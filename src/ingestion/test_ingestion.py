"""Test estrazione ingestion (G1/G2): markdown, PDF, immagini e video.

`tokenizer` è una fixture deterministica del modulo (contratto piano G1):
encode/decode a livello di carattere, round-trip esatto, conteggio token
controllato per i test di limite (1 char = 1 token).
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from src.contracts import MAX_SNIPPET_CHARS, make_memory_id
from src.ingestion.extractors import (
    InvalidMediaError,
    UnsupportedFormatError,
    extract_file_chunks,
)

MAX_PART_TOKENS = 1500
OVERLAP_TOKENS = 100


class CharTokenizer:
    """Tokenizer deterministica a livello di carattere per i test di limiti."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


@pytest.fixture
def tokenizer():
    return CharTokenizer()


# --------------------------------------------------------------------------- helper


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media_content_sha(source_sha: str, chunk_key: str) -> str:
    return hashlib.sha256((source_sha + chr(0) + chunk_key).encode("utf-8")).hexdigest()


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def assert_overlap(prev_text: str, next_text: str, tail_len: int = 50) -> None:
    """Consecutive parts of one section share ~100 tokens: the tail of the
    previous part reappears near the head of the next (offset 0 if prepended,
    up to `overlap` if the shared region is a hard-window boundary)."""
    tail = prev_text[-tail_len:]
    assert tail in next_text[: OVERLAP_TOKENS + tail_len + 4]


def assert_pair_invariants(pair, tokenizer, source_path: Path, source_sha: str) -> None:
    memory, request = pair
    # corpo completo nel payload, non solo il titolo
    assert memory.content_text == request.text_payload
    assert len(memory.snippet_text) <= MAX_SNIPPET_CHARS
    assert memory.file_path == str(source_path.resolve())
    assert memory.source_sha256 == source_sha
    assert memory.id == make_memory_id(str(source_path.resolve()), memory.chunk_key, memory.content_sha256)
    assert request.request_id
    assert not request.is_query
    # nessun chunk fuori limite
    if request.text_payload:
        assert token_count(tokenizer, request.text_payload) <= MAX_PART_TOKENS


# --------------------------------------------------------------------------- markdown (G1)


def test_markdown_body_is_embedded(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    source.write_text("# Cache\nLa cache conserva risultati per evitare ricalcoli.\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) == 1
    memory, request = pairs[0]
    assert "evitare ricalcoli" in request.text_payload
    assert memory.content_text == request.text_payload
    assert memory.snippet_text.startswith("La cache")
    assert memory.modality == "markdown"


def test_markdown_introduction_before_first_heading(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    source.write_text("Testo prima di qualsiasi titolo.\n\n# Cache\nCorpo della cache.\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert [m.chunk_key for m, _ in pairs] == [
        "section:Introduzione:0:part:0",
        "section:Cache:0:part:0",
    ]
    intro = pairs[0][0]
    assert intro.context_header == "f01.md > Introduzione"
    assert "Testo prima di qualsiasi titolo." in intro.content_text


def test_markdown_hierarchy_and_duplicate_headers(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    source.write_text(
        "# Guida\nIntroduzione della guida.\n## Cache\nPrima sezione cache.\n## Cache\nSeconda sezione cache.\n"
    )
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert [m.chunk_key for m, _ in pairs] == [
        "section:Guida:0:part:0",
        "section:Guida > Cache:0:part:0",
        "section:Guida > Cache:1:part:0",
    ]
    assert [m.context_header for m, _ in pairs] == [
        "f01.md > Guida",
        "f01.md > Guida > Cache",
        "f01.md > Guida > Cache",
    ]
    assert "Prima sezione cache." in pairs[1][0].content_text
    assert "Seconda sezione cache." in pairs[2][0].content_text


def test_markdown_heading_inside_code_fence_is_not_a_split(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    source.write_text("# Reale\n\n```python\n# Non un titolo\nx = 1\n```\n\nDopo il fence.\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert [m.chunk_key for m, _ in pairs] == ["section:Reale:0:part:0"]
    assert "# Non un titolo" in pairs[0][0].content_text
    assert "Dopo il fence." in pairs[0][0].content_text


def test_empty_markdown_has_no_chunks(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    source.write_text("")
    assert extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage")) == []


def test_markdown_long_section_splits_on_paragraphs_with_overlap(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    paragraphs = [f"Paragrafo numero {i} " + ("parola " * 95) + f"finep{i}." for i in range(4)]
    # 4 paragrafi da ~600 token: oltre il limite, confini di paragrafo disponibili
    source.write_text("# Lunga\n\n" + "\n\n".join(paragraphs) + "\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) >= 2
    for memory, _request in pairs:
        assert memory.chunk_key.startswith("section:Lunga:")
        assert token_count(tokenizer, memory.content_text) <= MAX_PART_TOKENS
    keys = [m.chunk_key for m, _ in pairs]
    assert keys == [f"section:Lunga:0:part:{i}" for i in range(len(pairs))]
    # overlap fra parti consecutive della stessa sezione: la parte successiva
    # inizia con la coda di 100 token della precedente
    for prev, nxt in zip(pairs, pairs[1:]):
        tail = prev[0].content_text[-OVERLAP_TOKENS:]
        assert nxt[0].content_text.startswith(tail)


def test_markdown_single_giant_paragraph_splits_by_tokens(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    body = " ".join(f"parola{i}" for i in range(1200))  # ~3600 token, un solo paragrafo
    source.write_text("# Muro\n\n" + body + "\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) >= 3
    for memory, _ in pairs:
        assert token_count(tokenizer, memory.content_text) <= MAX_PART_TOKENS
        assert memory.chunk_key.startswith("section:Muro:0:part:")
    # overlap token fra parti consecutive
    for prev, nxt in zip(pairs, pairs[1:]):
        assert_overlap(prev[0].content_text, nxt[0].content_text)


def test_markdown_no_overlap_across_sections(tmp_path, tokenizer):
    p_long = " ".join(f"parola{i}" for i in range(500))  # ~1500 token
    source = tmp_path / "f01.md"
    source.write_text(f"# Uno\n\n{p_long}\n\n# Due\n\nCorpo breve della sezione due.\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) >= 2
    assert pairs[-1][0].chunk_key.startswith("section:Due:")
    # la prima parte della sezione successiva non inizia con la coda della precedente
    assert not pairs[-1][0].content_text.startswith(pairs[-2][0].content_text[-OVERLAP_TOKENS:])


def test_markdown_identity_fields_and_content_hash(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    source.write_text("# Cache\r\nCorpo con CRLF.\r\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    memory, request = pairs[0]
    expected_body = "Corpo con CRLF."
    expected_sha = hashlib.sha256(f"Cache\n{expected_body}".encode("utf-8")).hexdigest()
    assert memory.content_sha256 == expected_sha
    assert memory.id == make_memory_id(str(source.resolve()), "section:Cache:0:part:0", expected_sha)
    assert memory.source_sha256 == sha256_file(source)
    assert memory.modality == "markdown"
    assert request.context_header == "f01.md > Cache"
    assert_pair_invariants(pairs[0], tokenizer, source, sha256_file(source))


def test_markdown_snippet_capped_at_350_chars(tmp_path, tokenizer):
    source = tmp_path / "f01.md"
    body = "x" * 900
    source.write_text(f"# Titolo\n\n{body}\n")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    memory, _ = pairs[0]
    assert memory.snippet_text == memory.content_text[:MAX_SNIPPET_CHARS]
    assert len(memory.snippet_text) <= MAX_SNIPPET_CHARS


def test_unsupported_extension_raises(tmp_path, tokenizer):
    source = tmp_path / "f01.txt"
    source.write_text("testo")
    with pytest.raises(UnsupportedFormatError):
        extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))


# --------------------------------------------------------------------------- PDF (G1)


def _write_png(path: Path, color=(200, 40, 40), size=(60, 40)) -> None:
    from PIL import Image

    Image.new("RGB", size, color).save(str(path))


def make_pdf(path: Path, pages: list[dict], page_size: tuple[int, int] = (842, 595), fontsize: int = 12) -> None:
    """Costruisce un PDF di prova: page = {'text': str | None, 'image': bool}."""
    import pymupdf

    doc = pymupdf.open()
    try:
        for spec in pages:
            page = doc.new_page(width=page_size[0], height=page_size[1])
            if spec.get("image"):
                png = path.with_suffix(".asset.png")
                _write_png(png)
                page.insert_image(pymupdf.Rect(100, 100, 320, 240), filename=str(png))
            if spec.get("text"):
                box = pymupdf.Rect(40, 40, page_size[0] - 40, page_size[1] - 40)
                unused = page.insert_textbox(box, spec["text"], fontname="helv", fontsize=fontsize)
                if unused < 0:
                    raise ValueError(f"il testo di prova non entra nella pagina: {unused}")
        doc.save(str(path))
    finally:
        doc.close()


def test_pdf_text_image_and_empty_pages(tmp_path, tokenizer):
    source = tmp_path / "g01.pdf"
    long_text = (
        "Questa pagina contiene testo sufficiente per essere indicizzata come pagina "
        "testuale completa, con oltre ottanta caratteri utili estratti dal documento."
    )
    make_pdf(
        source,
        [{"text": long_text}, {"image": True, "text": "Didascalia breve."}, {}],
    )
    stage = tmp_path / "stage"
    pairs = extract_file_chunks(str(source), tokenizer, str(stage))
    # pagina 3 completamente vuota: nessun record
    assert [m.page_num for m, _ in pairs] == [1, 2]

    text_memory, text_request = pairs[0]
    assert text_memory.modality == "pdf"
    assert text_memory.chunk_key == "page:1:part:0"
    assert text_memory.context_header == "g01.pdf [Pag. 1]"
    assert "pagina contiene testo sufficiente" in text_memory.content_text
    assert text_memory.content_text == text_request.text_payload
    expected_sha = hashlib.sha256(f"[Pag. 1]\n{text_memory.content_text}".encode("utf-8")).hexdigest()
    assert text_memory.content_sha256 == expected_sha

    visual_memory, visual_request = pairs[1]
    assert visual_memory.page_num == 2
    assert len(visual_request.image_paths) == 1
    png_path = Path(visual_request.image_paths[0])
    assert png_path.exists() and png_path.suffix == ".png"
    # lo snapshot sta in staging, non nei record persistenti come path temporaneo spurio
    assert str(stage) in str(png_path)
    assert visual_memory.snippet_text == "Pagina 2, contenuto visuale; allegato richiesto"
    assert visual_memory.source_sha256 == sha256_file(source)
    assert visual_memory.content_sha256 == media_content_sha(sha256_file(source), visual_memory.chunk_key)
    assert visual_memory.chunk_key.startswith("page:2:")
    for pair in pairs:
        assert_pair_invariants(pair, tokenizer, source, sha256_file(source))


def test_pdf_long_page_splits_with_parts(tmp_path, tokenizer):
    source = tmp_path / "g02.pdf"
    long_text = " ".join(f"parola{i}" for i in range(1200))
    make_pdf(source, [{"text": long_text}], page_size=(595, 3368), fontsize=8)
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) >= 2
    assert [m.chunk_key for m, _ in pairs] == [f"page:1:part:{i}" for i in range(len(pairs))]
    for memory, _ in pairs:
        assert token_count(tokenizer, memory.content_text) <= MAX_PART_TOKENS
    for prev, nxt in zip(pairs, pairs[1:]):
        assert_overlap(prev[0].content_text, nxt[0].content_text)


def test_pdf_short_page_without_images_renders_snapshot(tmp_path, tokenizer):
    source = tmp_path / "g03.pdf"
    make_pdf(source, [{"text": "Pochi caratteri."}])
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) == 1
    memory, request = pairs[0]
    assert len(request.image_paths) == 1
    assert Path(request.image_paths[0]).exists()
    assert memory.snippet_text == "Pagina 1, contenuto visuale; allegato richiesto"


def test_pdf_corrupt_file_raises_invalid_media(tmp_path, tokenizer):
    source = tmp_path / "g04.pdf"
    source.write_bytes(b"questo non e un pdf")
    with pytest.raises(InvalidMediaError):
        extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))


def test_pdf_render_respects_max_side(tmp_path, tokenizer):
    import pymupdf
    from PIL import Image

    source = tmp_path / "g05.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=2000, height=1400)  # a 120 dpi supera 1600 px
    page.insert_text((50, 60), "Poche parole.", fontname="helv", fontsize=20)
    doc.save(str(source))
    doc.close()
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) == 1
    request = pairs[0][1]
    assert len(request.image_paths) == 1
    with Image.open(request.image_paths[0]) as im:
        assert max(im.size) <= 1600


# --------------------------------------------------------------------------- immagini (G2)


def _write_test_png(path: Path, size=(640, 480), mode="RGB", color=(200, 40, 40)) -> None:
    from PIL import Image

    if mode == "RGBA":
        Image.new("RGBA", size, color + (128,)).save(str(path))
    else:
        Image.new("RGB", size, color).save(str(path))


def test_png_image_single_chunk(tmp_path, tokenizer):
    source = tmp_path / "f09.png"
    _write_test_png(source, size=(640, 480))
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert len(pairs) == 1
    memory, request = pairs[0]
    assert memory.modality == "image"
    assert memory.chunk_key == "image:0"
    assert memory.context_header == "f09.png > immagine"
    assert memory.snippet_text == "Immagine 640x480"
    assert memory.content_text == ""
    assert request.text_payload == ""
    assert request.image_paths == [str(source.resolve())]  # RGB: nessuna copia di staging
    sha = sha256_file(source)
    assert memory.source_sha256 == sha
    assert memory.content_sha256 == media_content_sha(sha, "image:0")
    assert memory.id == make_memory_id(str(source.resolve()), "image:0", memory.content_sha256)


def test_rgba_image_converted_to_rgb_in_staging(tmp_path, tokenizer):
    from PIL import Image

    source = tmp_path / "f10.png"
    _write_test_png(source, size=(300, 200), mode="RGBA")
    stage = tmp_path / "stage"
    pairs = extract_file_chunks(str(source), tokenizer, str(stage))
    assert len(pairs) == 1
    request = pairs[0][1]
    assert len(request.image_paths) == 1
    copy_path = Path(request.image_paths[0])
    assert copy_path != source.resolve()
    assert str(stage) in str(copy_path) and copy_path.exists()
    with Image.open(copy_path) as im:
        assert im.mode == "RGB"
        assert im.size == (300, 200)


def test_corrupt_image_raises_invalid_media(tmp_path, tokenizer):
    source = tmp_path / "f11.png"
    source.write_bytes(b"questo non e un immagine")
    with pytest.raises(InvalidMediaError):
        extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))


# --------------------------------------------------------------------------- video (G2)


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="richiede ffmpeg/ffprobe",
)


def make_video(path: Path, duration: float, with_audio: bool = True) -> None:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=128x96:rate=2",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-an"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)


@needs_ffmpeg
def test_video_95s_produces_three_windows(tmp_path, tokenizer):
    source = tmp_path / "f13.mp4"
    make_video(source, 95.0, with_audio=True)
    stage = tmp_path / "stage"
    pairs = extract_file_chunks(str(source), tokenizer, str(stage))
    assert [(m.start_sec, m.end_sec) for m, _ in pairs] == [(0.0, 45.0), (40.0, 85.0), (80.0, 95.0)]
    assert [m.chunk_key for m, _ in pairs] == [
        "video:0:45000",
        "video:40000:85000",
        "video:80000:95000",
    ]
    source_sha = sha256_file(source)
    for memory, request in pairs:
        assert memory.modality == "video"
        assert memory.context_header == f"f13.mp4 > [{memory.start_sec:.1f}s, {memory.end_sec:.1f}s)"
        assert memory.snippet_text == (f"Video 95 s, finestra [{memory.start_sec:.0f}, {memory.end_sec:.0f})")
        assert memory.content_text == "" and request.text_payload == ""
        assert len(request.image_paths) == 3
        for frame in request.image_paths:
            frame_path = Path(frame)
            assert frame_path.exists() and frame_path.suffix == ".png"
            assert str(stage) in str(frame_path)
        assert request.audio_path is not None
        assert Path(request.audio_path).exists()
        import wave

        with wave.open(request.audio_path, "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
        assert memory.content_sha256 == media_content_sha(source_sha, memory.chunk_key)
        assert memory.id == make_memory_id(str(source.resolve()), memory.chunk_key, memory.content_sha256)
        assert_pair_invariants((memory, request), tokenizer, source, source_sha)


@needs_ffmpeg
def test_video_without_audio_has_null_audio_path(tmp_path, tokenizer):
    source = tmp_path / "f14.mp4"
    make_video(source, 45.0, with_audio=False)
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert [(m.start_sec, m.end_sec) for m, _ in pairs] == [(0.0, 45.0)]
    assert all(request.audio_path is None for _m, request in pairs)
    assert all(len(request.image_paths) == 3 for _m, request in pairs)


@needs_ffmpeg
def test_very_short_video_keeps_one_window_with_frames(tmp_path, tokenizer):
    source = tmp_path / "f15.mp4"
    make_video(source, 2.0, with_audio=True)
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert [(m.start_sec, m.end_sec) for m, _ in pairs] == [(0.0, 2.0)]
    for _memory, request in pairs:
        assert len(request.image_paths) == 3
        assert all(Path(frame).exists() for frame in request.image_paths)


@needs_ffmpeg
def test_corrupt_video_raises_invalid_media(tmp_path, tokenizer):
    source = tmp_path / "f16.mp4"
    source.write_bytes(b"questo non e un video")
    with pytest.raises(InvalidMediaError):
        extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))


# --------------------------------------------------------------------------- finestre video (pure)


def test_video_windows_boundaries():
    from src.ingestion.extractors import _video_windows

    assert _video_windows(95.0) == [(0.0, 45.0), (40.0, 85.0), (80.0, 95.0)]
    assert _video_windows(45.0) == [(0.0, 45.0)]
    assert _video_windows(2.0) == [(0.0, 2.0)]
    # padding del contenitore: nessuna finestra spuria di coda
    assert _video_windows(45.04) == [(0.0, 45.04)]
    assert _video_windows(90.0) == [(0.0, 45.0), (40.0, 85.0), (80.0, 90.0)]


def test_audio_window_capped_at_wire_limit(tmp_path, tokenizer, monkeypatch):
    """Finestra che assorbe il padding del contenitore (es. [0, 45.04)): il WAV
    richiesto al modulo media resta entro il limite wire dei 45 s del contratto
    (canary su backend reale 2026-10-18: INPUT_TOO_LARGE audio 45.0s)."""
    import src.ingestion.extractors as extractors

    recorded = {}

    def fake_probe(path):
        return {
            "format": {"duration": "45.04"},
            "streams": [
                {"codec_type": "video", "duration": "45.04"},
                {"codec_type": "audio"},
            ],
        }

    def fake_frame(path, at_sec, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"png")
        return out

    def fake_audio(path, start, end, out):
        recorded["span"] = (start, end)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"wav")
        return out

    monkeypatch.setattr(extractors, "probe_video", fake_probe)
    monkeypatch.setattr(extractors, "extract_frame", fake_frame)
    monkeypatch.setattr(extractors, "extract_audio_window", fake_audio)
    source = tmp_path / "f17.mp4"
    source.write_bytes(b"payload")
    pairs = extract_file_chunks(str(source), tokenizer, str(tmp_path / "stage"))
    assert [(m.start_sec, m.end_sec) for m, _ in pairs] == [(0.0, 45.04)]
    assert recorded["span"] == (0.0, 45.0)  # cap al limite wire, non 45.04
