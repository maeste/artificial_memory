"""Estrattori di contenuto (piano Ingestion G1/G2): dispatch formati, chunk
testuali e metadati secondo il contratto 1.0 (07_TECHNICAL_CONTRACTS §3/§4).

Regole identità (contratto §3 «Identità dei file e dei chunk»):
- source_sha256 = SHA-256 dei byte originali dell'intero file;
- per testo (markdown/PDF testuali): content_sha256 = SHA-256 di UTF-8 di
  header logico senza nome file + newline + corpo, con solo CRLF normalizzato;
- per media (immagini, PDF visuali, video): content_sha256 = SHA-256 di
  source_sha256 + chr(0) + chunk_key;
- l'ID arriva sempre da make_memory_id su path canonico.

Limiti: sezioni/pagine oltre 1500 token vengono frazionate su confini di
paragrafo (o per token se un paragrafo è un muro), con overlap di 100 token
fra parti della stessa sezione e mai attraverso sezioni diverse. Nessun taglio
a byte: le parti sono assemblate a livello di token ID.
"""

import hashlib
import re
import time
from pathlib import Path

from src.contracts import (
    MAX_CONTEXT_HEADER_CHARS,
    MAX_SNIPPET_CHARS,
    EmbedRequest,
    MemoryView,
    make_memory_id,
)
from src.ingestion.media import (
    MediaDecodeError,
    decode_image,
    extract_audio_window,
    extract_frame,
    has_audio_stream,
    probe_video,
    render_page_png,
    save_rgb_copy,
    video_duration,
)

MAX_PART_TOKENS = 1500
OVERLAP_TOKENS = 100
PACK_BUDGET_TOKENS = MAX_PART_TOKENS - OVERLAP_TOKENS  # 1400: lascia spazio all'overlap
MIN_VISUAL_TEXT_CHARS = 80  # sotto questa soglia la pagina PDF viene resa come immagine
VIDEO_WINDOW_SEC = 45.0
VIDEO_STRIDE_SEC = 40.0
VIDEO_TAIL_EPSILON_SEC = 0.5  # code oltre questo sono contenuto reale, non padding

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
PDF_SUFFIXES = frozenset({".pdf"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
VIDEO_SUFFIXES = frozenset({".mp4"})
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES | VIDEO_SUFFIXES


class IngestionError(Exception):
    """Errore di estrazione con codice stabile per i risultati di scan."""

    error_code = "INGESTION_ERROR"


class UnsupportedFormatError(IngestionError):
    error_code = "UNSUPPORTED_FORMAT"


class InvalidMediaError(IngestionError):
    error_code = "INVALID_MEDIA"


# --------------------------------------------------------------------------- helper digest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_crlf(text: str) -> str:
    return text.replace("\r\n", "\n")


def _text_content_sha(logical_header: str, body: str) -> str:
    return _sha256_bytes(f"{_normalize_crlf(logical_header)}\n{_normalize_crlf(body)}".encode("utf-8"))


def _media_content_sha(source_sha: str, chunk_key: str) -> str:
    return _sha256_bytes(f"{source_sha}{chr(0)}{chunk_key}".encode("utf-8"))


def _snippet(text: str) -> str:
    return text[:MAX_SNIPPET_CHARS]


def _truncate_header(text: str) -> str:
    return text[:MAX_CONTEXT_HEADER_CHARS]


# --------------------------------------------------------------------------- costruzione record


def _text_pair(
    path: Path,
    source_sha: str,
    modality: str,
    chunk_key: str,
    display_header: str,
    logical_header: str,
    body: str,
    page_num: int | None = None,
) -> tuple[MemoryView, EmbedRequest]:
    content_sha = _text_content_sha(logical_header, body)
    memory = MemoryView(
        id=make_memory_id(str(path), chunk_key, content_sha),
        file_path=str(path),
        modality=modality,  # type: ignore[arg-type]
        chunk_key=chunk_key,
        content_sha256=content_sha,
        source_sha256=source_sha,
        context_header=display_header,
        snippet_text=_snippet(body),
        content_text=body,
        page_num=page_num,
        created_at=time.time(),
    )
    request = EmbedRequest(
        request_id=f"ing:{source_sha[:16]}:{chunk_key}",
        is_query=False,
        text_payload=body,
        context_header=display_header,
    )
    return memory, request


def _media_pair(
    path: Path,
    source_sha: str,
    modality: str,
    chunk_key: str,
    display_header: str,
    *,
    body: str = "",
    snippet: str = "",
    image_paths: list[str] | None = None,
    audio_path: str | None = None,
    page_num: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[MemoryView, EmbedRequest]:
    content_sha = _media_content_sha(source_sha, chunk_key)
    memory = MemoryView(
        id=make_memory_id(str(path), chunk_key, content_sha),
        file_path=str(path),
        modality=modality,  # type: ignore[arg-type]
        chunk_key=chunk_key,
        content_sha256=content_sha,
        source_sha256=source_sha,
        context_header=display_header,
        snippet_text=_snippet(snippet or body),
        content_text=body,
        page_num=page_num,
        start_sec=start_sec,
        end_sec=end_sec,
        created_at=time.time(),
    )
    request = EmbedRequest(
        request_id=f"ing:{source_sha[:16]}:{chunk_key}",
        is_query=False,
        text_payload=body,
        context_header=display_header,
        image_paths=list(image_paths or []),
        audio_path=audio_path,
    )
    return memory, request


# --------------------------------------------------------------------------- chunking testuale


def _paragraphs(raw: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n[ \t]*\n", raw) if block.strip()]


def _hard_windows(ids: list[int]) -> list[list[int]]:
    """Finestre di token da 1400 con stride 1300: overlap di 100 fra finestre."""
    stride = PACK_BUDGET_TOKENS - OVERLAP_TOKENS
    windows: list[list[int]] = []
    start = 0
    while start < len(ids):
        end = min(start + PACK_BUDGET_TOKENS, len(ids))
        windows.append(ids[start:end])
        if end >= len(ids):
            break
        start += stride
    return windows


def _split_into_parts(body: str, tokenizer: object) -> list[str]:
    """Fraziona un corpo oltre il limite su confini di paragrafo.

    Un paragrafo oltre il budget viene diviso per token (finestre con overlap
    interno). Le parti consecutive della stessa sezione condividono 100 token:
    la parte inizia con la coda di token della precedente, mai attraverso
    sezioni diverse. L'assemblaggio avviene a livello di token ID e la dimensione
    massima è assoluta: se l'overlap non entra viene accorciato dall'inizio.
    """
    paragraphs = _paragraphs(body)
    if not paragraphs:
        return []
    # blocchi: (token_ids, continua_il_precedente) — le finestre di uno stesso
    # paragrafo oltre budget portano già il loro overlap interno
    blocks: list[tuple[list[int], bool]] = []
    for paragraph in paragraphs:
        ids = list(tokenizer.encode(paragraph, add_special_tokens=False))  # type: ignore[attr-defined]
        if len(ids) <= PACK_BUDGET_TOKENS:
            blocks.append((ids, False))
        else:
            for index, window in enumerate(_hard_windows(ids)):
                blocks.append((window, index > 0))

    # packing: i blocchi paragrafo si accampano fino al budget; una finestra
    # oltre-budget apre sempre una parte propria (il suo overlap è interno)
    parts: list[list[list[int]]] = []
    first_block_continues: list[bool] = []
    current: list[list[int]] = []
    current_tokens = 0
    current_continues = False
    for ids, continues in blocks:
        if current and current_tokens + len(ids) > PACK_BUDGET_TOKENS:
            parts.append(current)
            first_block_continues.append(current_continues)
            current, current_tokens = [], 0
        if not current:
            current_continues = continues
        current.append(ids)
        current_tokens += len(ids)
    if current:
        parts.append(current)
        first_block_continues.append(current_continues)

    separator_ids = list(tokenizer.encode("\n\n", add_special_tokens=False))  # type: ignore[attr-defined]
    assembled: list[str] = []
    previous_ids: list[int] | None = None
    for part, starts_continuation in zip(parts, first_block_continues, strict=True):
        content_ids: list[int] = []
        for ids in part:
            content_ids.extend(ids)
        if previous_ids is None or starts_continuation:
            final_ids = content_ids
        else:
            tail = previous_ids[-OVERLAP_TOKENS:]
            final_ids = tail + separator_ids + content_ids
            if len(final_ids) > MAX_PART_TOKENS:
                final_ids = final_ids[len(final_ids) - MAX_PART_TOKENS :]
        assembled.append(tokenizer.decode(final_ids))  # type: ignore[attr-defined]
        previous_ids = final_ids
    return assembled


# --------------------------------------------------------------------------- markdown


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

INTRODUCTION_TITLE = "Introduzione"


def _parse_markdown_sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Sezioni (gerarchia titoli, corpo) split per heading ATX fuori da code fence.

    Il testo prima del primo heading è la sezione «Introduzione». Le sezioni
    senza corpo vengono scartate: il titolo non sostituisce il contenuto.
    """
    sections: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    pending: list[str] = []
    fence: str | None = None

    def close() -> None:
        body = "\n\n".join(_paragraphs("\n".join(pending)))
        pending.clear()
        if not body:
            return
        hierarchy = tuple(title for _level, title in stack) if stack else (INTRODUCTION_TITLE,)
        sections.append((hierarchy, body))

    for line in text.split("\n"):
        stripped = line.strip()
        if fence is not None:
            pending.append(line)
            if stripped.startswith(fence):
                fence = None
            continue
        opening = _FENCE_RE.match(stripped)
        if opening:
            fence = opening.group(1)[0] * 3
            pending.append(line)
            continue
        heading = _HEADING_RE.match(stripped)
        if heading:
            close()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            continue
        pending.append(line)
    close()
    return sections


def _extract_markdown(path: Path, tokenizer: object) -> list[tuple[MemoryView, EmbedRequest]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise InvalidMediaError(f"Markdown non leggibile: {path}: {exc}") from exc
    source_sha = _sha256_bytes(path.read_bytes())
    occurrences: dict[tuple[str, ...], int] = {}
    pairs: list[tuple[MemoryView, EmbedRequest]] = []
    for hierarchy, body in _parse_markdown_sections(text):
        occurrence = occurrences.get(hierarchy, 0)
        occurrences[hierarchy] = occurrence + 1
        hierarchy_str = " > ".join(hierarchy)
        display = _truncate_header(f"{path.name} > {hierarchy_str}")
        for part_index, part_body in enumerate(_split_into_parts(body, tokenizer)):
            chunk_key = f"section:{hierarchy_str}:{occurrence}:part:{part_index}"
            pairs.append(_text_pair(path, source_sha, "markdown", chunk_key, display, hierarchy_str, part_body))
    return pairs


# --------------------------------------------------------------------------- PDF


def _page_text_blocks(page) -> list[str]:
    """Blocchi di testo della pagina in ordine di lettura, righe ricongiunte."""
    blocks = [block for block in page.get_text("blocks") if block[6] == 0 and block[4].strip()]
    blocks.sort(key=lambda block: (round(block[1]), block[0]))
    return [" ".join(block[4].split()) for block in blocks]


def _extract_pdf(path: Path, tokenizer: object, staging: Path) -> list[tuple[MemoryView, EmbedRequest]]:
    import pymupdf

    source_sha = _sha256_bytes(path.read_bytes())
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise InvalidMediaError(f"PDF non leggibile: {path}: {exc}") from exc
    pairs: list[tuple[MemoryView, EmbedRequest]] = []
    try:
        for index in range(doc.page_count):
            page = doc[index]
            page_num = index + 1  # le pagine PDF sono numerate da 1
            body = "\n\n".join(_page_text_blocks(page))
            useful_chars = len(re.sub(r"\s+", "", body))
            has_images = bool(page.get_images(full=True))
            if useful_chars == 0 and not has_images:
                continue  # pagina completamente vuota senza immagini: nessun record
            display = _truncate_header(f"{path.name} [Pag. {page_num}]")
            logical = f"[Pag. {page_num}]"
            if has_images or useful_chars < MIN_VISUAL_TEXT_CHARS:
                # pagina visuale: snapshot in staging, testo e immagine insieme
                png = render_page_png(page, staging, page_num)
                snippet = (
                    body[:MAX_SNIPPET_CHARS]
                    if useful_chars >= MIN_VISUAL_TEXT_CHARS
                    else f"Pagina {page_num}, contenuto visuale; allegato richiesto"
                )
                pairs.append(
                    _media_pair(
                        path,
                        source_sha,
                        "pdf",
                        f"page:{page_num}:part:0",
                        display,
                        body=body,
                        snippet=snippet,
                        image_paths=[str(png)],
                        page_num=page_num,
                    )
                )
                continue
            for part_index, part_body in enumerate(_split_into_parts(body, tokenizer)):
                chunk_key = f"page:{page_num}:part:{part_index}"
                pairs.append(_text_pair(path, source_sha, "pdf", chunk_key, display, logical, part_body, page_num))
    finally:
        doc.close()
    return pairs


# --------------------------------------------------------------------------- immagini


def _extract_image(path: Path, staging: Path) -> list[tuple[MemoryView, EmbedRequest]]:
    width, height, mode = decode_image(path)
    if mode == "RGB":
        image_paths = [str(path)]  # decode valido: la richiesta usa l'originale
    else:
        image_paths = [str(save_rgb_copy(path, staging))]  # solo la copia di staging
    return [
        _media_pair(
            path,
            _sha256_bytes(path.read_bytes()),
            "image",
            "image:0",
            _truncate_header(f"{path.name} > immagine"),
            snippet=f"Immagine {width}x{height}",
            image_paths=image_paths,
        )
    ]


# --------------------------------------------------------------------------- MP4


def _video_windows(duration: float) -> list[tuple[float, float]]:
    """Finestre di 45 s con stride 40 s, l'ultima troncata alla fine.

    Una coda residua sotto la epsilon è padding del contenitore, non contenuto:
    viene assorbita nell'ultima finestra invece di generare una finestra spuria.
    """
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(start + VIDEO_WINDOW_SEC, duration)
        if duration - end < VIDEO_TAIL_EPSILON_SEC:
            end = duration
        windows.append((start, end))
        if end >= duration:
            break
        start += VIDEO_STRIDE_SEC
    return windows


def _frame_or_fallback(path: Path, at_sec: float, duration: float, out: Path) -> Path:
    """Frame alla posizione richiesta; per video brevi retrocede sulla scala
    (metà durata, poi 0) finché non trova un frame valido: la finestra resta
    con almeno un frame invece di fallire."""
    candidates = list(dict.fromkeys((at_sec, duration / 2.0, 0.0)))
    for candidate in candidates:
        try:
            return extract_frame(path, candidate, out)
        except MediaDecodeError:
            continue
    raise InvalidMediaError(f"Nessun frame estraibile da {path.name}")


def _extract_video(path: Path, tokenizer: object, staging: Path) -> list[tuple[MemoryView, EmbedRequest]]:
    probe = probe_video(path)
    duration = video_duration(probe)
    if duration <= 0:
        raise InvalidMediaError(f"Durata video non positiva: {path.name}")
    with_audio = has_audio_stream(probe)
    source_sha = _sha256_bytes(path.read_bytes())
    pairs: list[tuple[MemoryView, EmbedRequest]] = []
    for window_index, (start, end) in enumerate(_video_windows(duration)):
        frame_paths = [
            str(
                _frame_or_fallback(
                    path,
                    start + (end - start) * fraction,  # 10%, 50%, 90% della finestra
                    duration,
                    staging / "frames" / f"w{window_index}_f{frame_index}.png",
                )
            )
            for frame_index, fraction in enumerate((0.1, 0.5, 0.9))
        ]
        audio_path: str | None = None
        if with_audio:
            # il WAV rispetta il limite wire di 45 s del contratto anche quando
            # la finestra assorbe il padding del contenitore (es. 45.04 s):
            # la copertura visuale resta intera, l'audio cede solo la coda.
            wav = extract_audio_window(
                path,
                start,
                min(end, start + VIDEO_WINDOW_SEC),
                staging / "audio" / f"w{window_index}.wav",
            )
            audio_path = str(wav)
        chunk_key = f"video:{int(round(start * 1000))}:{int(round(end * 1000))}"
        display = _truncate_header(f"{path.name} > [{start:.1f}s, {end:.1f}s)")
        snippet = f"Video {duration:.0f} s, finestra [{start:.0f}, {end:.0f})"
        pairs.append(
            _media_pair(
                path,
                source_sha,
                "video",
                chunk_key,
                display,
                snippet=snippet,
                image_paths=frame_paths,
                audio_path=audio_path,
                start_sec=start,
                end_sec=end,
            )
        )
    return pairs


# --------------------------------------------------------------------------- dispatch


def extract_file_chunks(file_path: str, tokenizer: object, temp_dir: str) -> list[tuple[MemoryView, EmbedRequest]]:
    """Estrae i chunk (MemoryView, EmbedRequest) di un file supportato.

    Solo estrazione: i temporanei in temp_dir restano validi fino al termine
    dell'indicizzazione (li chiude index_file). Il path canonico è
    Path.resolve(); i formati non supportati sono errore esplicito.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise InvalidMediaError(f"File sorgente inesistente: {path}")
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return _extract_markdown(path, tokenizer)
    staging = Path(temp_dir)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        if suffix in PDF_SUFFIXES:
            return _extract_pdf(path, tokenizer, staging)
        if suffix in IMAGE_SUFFIXES:
            return _extract_image(path, staging)
        if suffix in VIDEO_SUFFIXES:
            return _extract_video(path, tokenizer, staging)
    except MediaDecodeError as exc:
        raise InvalidMediaError(str(exc)) from exc
    raise UnsupportedFormatError(f"Formato non supportato: {suffix!r} ({path.name})")
