"""Selezione contenuti, budget, staging e pacchetto di contesto (piano HUD H2).

L'exporter lavora solo su uno snapshot coerente (LiveStateEvent + session_inputs
dalla risposta controllo), non su liste UI. Il pacchetto nasce in staging e
arriva alla destinazione finale con rename atomico; un errore fatale rimuove
lo staging e solleva EXPORT_FAILED, senza pacchetti parziali.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from src.contracts import (
    SCHEMA_VERSION,
    ExportResult,
    ExportSettings,
    LiveInput,
    LiveStateEvent,
    MemoryView,
    SearchSettings,
    SpaceSpec,
)

BUDGET_MIN_CHARS = 12_000
RECENT_AGENT_TEXT_COUNT = 5
RECENT_MIC_WINDOW_SEC = 60.0
PIN_EXTRACT_CHARS = 1_500
TITLE_MAX_CHARS = 120
MAX_OTHER_SUGGESTIONS = 3
MOCK_WARNING = (
    "> Avvertenza: backend mock — dati sperimentali non semantici, non provano la qualità del retrieval reale."
)
SOURCE_UNAVAILABLE = "originale selezionato indisponibile"
TEXT_ONLY_LIMITATION = (
    "Fonte immagine o pagina scansionata: un agente solo testuale non ne legge il contenuto; "
    "nessuna caption automatica è promessa."
)
OPERATIONAL_LIMITS = [
    "PoC: le copie di staging richiedono spazio disco pari alle sorgenti selezionate.",
]
MAX_LEASE_ATTEMPTS = 3


class ExportFailedError(RuntimeError):
    """Errore fatale dell'export: nessun pacchetto pubblicato."""


class _LeaseExpired(Exception):
    """Il lease è scaduto durante l'acquisizione: ripartire da un nuovo snapshot."""


@dataclass(frozen=True)
class _PinView:
    """Rappresentazione di un pin pronta per il Markdown."""

    title: str
    memory_id: str
    source_label: str
    selection_label: str
    extract: str
    text_relpath: str


# --------------------------------------------------------------------------- selezione


def _select_inputs(state: LiveStateEvent, session_inputs: list[LiveInput]) -> tuple[list[LiveInput], list[str]]:
    """Ultimi cinque agent_text, mic entro 60 s e input referenziati dai pin.

    Ritorna gli input in ordine di storia (deduplicati per ID) e gli ID dei
    pin assenti dalla storia.
    """
    by_id: dict[str, LiveInput] = {}
    for item in session_inputs:
        by_id.setdefault(item.id, item)
    agent_recent = [item for item in session_inputs if item.source == "agent_text"][-RECENT_AGENT_TEXT_COUNT:]
    mic_window = [
        item
        for item in session_inputs
        if item.source == "mic_audio" and item.occurred_at >= state.emitted_at - RECENT_MIC_WINDOW_SEC
    ]
    selected_ids = {item.id for item in agent_recent} | {item.id for item in mic_window}
    missing: list[str] = []
    for pin in state.pinned:
        for input_id in pin.snapshot.input_ids:
            if input_id in by_id:
                selected_ids.add(input_id)
            elif input_id not in missing:
                missing.append(input_id)  # ordine preservato, senza duplicati fra pin
    ordered = [item for item in session_inputs if item.id in selected_ids]
    return ordered, missing


# --------------------------------------------------------------------------- acquisizione audio


def _acquire_audio(
    selected: list[LiveInput],
    staging: Path,
    *,
    lease_expires_at: float | None,
    clock,
) -> tuple[dict[str, str], list[str]]:
    """Copia i WAV selezionati in assets con nomi UUID, dai descrittori aperti.

    Con un lease: apre tutti gli handle entro la scadenza e ricontrolla il
    tempo dopo l'apertura; se scaduto chiude tutto e segnala _LeaseExpired
    (ripartenza da un nuovo snapshot, senza pacchetto parziale). Su Linux
    gli handle aperti restano validi anche se un reset rimuove i path.
    """
    if lease_expires_at is not None and clock() > lease_expires_at:
        raise _LeaseExpired
    targets: list[tuple[LiveInput, object]] = []
    warnings: list[str] = []
    try:
        for item in selected:
            if item.source != "mic_audio":
                continue
            if not item.audio_path or not os.path.isfile(item.audio_path):
                warnings.append(f"audio {item.id} non disponibile sul runtime: {item.audio_path}")
                continue
            targets.append((item, open(item.audio_path, "rb")))
        if lease_expires_at is not None and clock() > lease_expires_at:
            raise _LeaseExpired
        assets: dict[str, str] = {}
        for item, handle in targets:
            rel = f"assets/{uuid.uuid4().hex}.wav"
            with open(staging / rel, "wb") as out:
                shutil.copyfileobj(handle, out)
            assets[item.id] = rel
        return assets, warnings
    finally:
        for _, handle in targets:
            handle.close()


# --------------------------------------------------------------------------- pin e sorgenti


def _copy_and_digest(source: Path, destination: Path, chunk_size: int = 1 << 20) -> str:
    """Copia in staging calcolando lo SHA-256 sui byte copiati."""
    digest = hashlib.sha256()
    with open(source, "rb") as src, open(destination, "wb") as dst:
        while chunk := src.read(chunk_size):
            digest.update(chunk)
            dst.write(chunk)
    return digest.hexdigest()


def _source_label(memory: MemoryView) -> str:
    label = f"{Path(memory.file_path).name} · {memory.modality}"
    if memory.page_num is not None:
        label += f" · pagina {memory.page_num}"
    if memory.start_sec is not None and memory.end_sec is not None:
        label += f" · intervallo [{memory.start_sec}, {memory.end_sec}) s"
    return label


def _attachment_entry(staging: Path, rel: str, kind: str, source_path: str) -> dict:
    return {
        "path": rel,
        "sha256": hashlib.sha256((staging / rel).read_bytes()).hexdigest(),
        "kind": kind,
        "source_path": source_path,
    }


def _render_pdf_page(source_copy: Path, page_num: int | None, staging: Path) -> list[str]:
    """Renderizza la pagina selezionata del PDF verificato in un PNG di assets."""
    import pymupdf  # import lazy: serve solo all'export con PDF

    doc = pymupdf.open(str(source_copy))
    try:
        if page_num is None or not 1 <= page_num <= doc.page_count:
            return []
        pix = doc.load_page(page_num - 1).get_pixmap(dpi=110)
        rel = f"assets/{uuid.uuid4().hex}.png"
        pix.save(str(staging / rel))
        return [rel]
    finally:
        doc.close()


def _extract_video_clip(source_copy: Path, start_sec: float, end_sec: float, destination: Path) -> None:
    """Estrae solo l'intervallo selezionato con seek accurato e re-encode.

    La clip esportata parte da zero; gli estremi originali restano nel
    manifest. Con re-encode la traccia audio, se presente, è conservata;
    un video senza audio resta clip visuale.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ExportFailedError("EXPORT_FAILED: ffmpeg non disponibile per l'estrazione video")
    duration = max(end_sec - start_sec, 0.1)
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source_copy),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ExportFailedError(f"EXPORT_FAILED: estrazione video non riuscita: {proc.stderr.strip()[:400]}")


def _produce_media_attachments(memory: MemoryView, staging_copy: Path, staging: Path) -> tuple[list[dict], list[str]]:
    """Allegati dalla copia verificata; ritorna (entries, limitazioni_markdown)."""
    entries: list[dict] = []
    limitations: list[str] = []
    if memory.modality == "image":
        suffix = Path(memory.file_path).suffix or ".png"
        rel = f"assets/{uuid.uuid4().hex}{suffix}"
        shutil.copyfile(staging_copy, staging / rel)
        entries.append(_attachment_entry(staging, rel, "image", memory.file_path))
        limitations.append(TEXT_ONLY_LIMITATION)
    elif memory.modality == "pdf":
        for rel in _render_pdf_page(staging_copy, memory.page_num, staging):
            entries.append(_attachment_entry(staging, rel, "pdf_page", memory.file_path))
        if not memory.content_text:
            limitations.append(TEXT_ONLY_LIMITATION)
    elif memory.modality == "video" and memory.start_sec is not None and memory.end_sec is not None:
        rel = f"assets/{uuid.uuid4().hex}.mp4"
        _extract_video_clip(staging_copy, memory.start_sec, memory.end_sec, staging / rel)
        entries.append(_attachment_entry(staging, rel, "video_clip", memory.file_path))
    return entries, limitations


def _export_pinned(
    state: LiveStateEvent, staging: Path
) -> tuple[list[_PinView], list[dict], dict[str, list[str]], list[dict], list[str], list[str]]:
    """Testi congelati in memories/, sorgenti verificate, allegati e viste.

    Tutti i render/tagli avvengono dalla copia di staging con digest
    verificato; la copia completa è rimossa dopo gli estratti. Ritorna
    (views, entries, memory_map, attachments, warnings, limitations).
    """
    views: list[_PinView] = []
    entries: list[dict] = []
    memory_map: dict[str, list[str]] = {}
    attachments: list[dict] = []
    warnings: list[str] = []
    limitations: list[str] = []
    for pin in state.pinned:
        memory = pin.memory
        text_rel = f"memories/{memory.id}.txt"
        (staging / text_rel).write_text(memory.content_text, encoding="utf-8")
        memory_map.setdefault(memory.id, []).append(text_rel)
        source_state = "verified"
        pin_attachments: list[dict] = []
        source = Path(memory.file_path)
        if not source.is_file():
            source_state = "unavailable"
            warnings.append(f"{SOURCE_UNAVAILABLE} per {memory.id}: file sorgente assente")
        else:
            staging_copy = staging / f"_source-{memory.id[:16]}"
            try:
                digest = _copy_and_digest(source, staging_copy)
                if digest != memory.source_sha256:
                    source_state = "unavailable"
                    warnings.append(f"{SOURCE_UNAVAILABLE} per {memory.id}: digest della sorgente diverso")
                else:
                    pin_attachments, pin_limitations = _produce_media_attachments(memory, staging_copy, staging)
                    limitations.extend(pin_limitations)
            finally:
                staging_copy.unlink(missing_ok=True)
        for attachment in pin_attachments:
            memory_map.setdefault(memory.id, []).append(attachment["path"])
            attachments.append(attachment)
        selection = ", ".join(pin.snapshot.input_ids) if pin.snapshot.input_ids else "nessuno"
        view = _PinView(
            title=memory.context_header[:TITLE_MAX_CHARS],
            memory_id=memory.id,
            source_label=_source_label(memory),
            selection_label=f"input snapshot: {selection} (pin del {pin.pinned_at:.0f})",
            extract=memory.content_text[:PIN_EXTRACT_CHARS],
            text_relpath=text_rel,
        )
        views.append(view)
        entries.append(
            {
                "memory_id": memory.id,
                "title": view.title,
                "source_path": memory.file_path,
                "source_sha256": memory.source_sha256,
                "modality": memory.modality,
                "page_num": memory.page_num,
                "start_sec": memory.start_sec,
                "end_sec": memory.end_sec,
                "pinned_at": pin.pinned_at,
                "snapshot_id": pin.snapshot.id,
                "snapshot_input_ids": list(pin.snapshot.input_ids),
                "text_file": text_rel,
                "source_state": source_state,
                "attachments": pin_attachments,
            }
        )
    return views, entries, memory_map, attachments, warnings, limitations


# --------------------------------------------------------------------------- markdown


def _fit_paragraphs(text: str, max_chars: int) -> tuple[str, bool]:
    """Riduce ai paragrafi interi che entrano nel budget; ritorna (testo, troncato)."""
    if len(text) <= max_chars:
        return text, False
    fitted: list[str] = []
    used = 0
    for paragraph in text.split("\n\n"):
        extra = len(paragraph) + (2 if fitted else 0)
        if used + extra > max_chars:
            break
        fitted.append(paragraph)
        used += extra
    if not fitted:
        return "", True
    return "\n\n".join(fitted), True


def _build_markdown(
    *,
    backend: str,
    session_id: str,
    pinned: list[_PinView],
    recent_inputs: list[LiveInput],
    suggestions: list,
    budget_chars: int,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    mic_transcripts: dict[str, str] | None = None,
    clip_transcripts: dict[str, str] | None = None,
    transcription_mode: str = "off",
) -> tuple[str, list[str]]:
    """Markdown del contesto dentro il budget, con priorità su paragrafi interi.

    Priorità: metadati/avvisi, titoli e riferimenti di tutti i pin (obbligatori:
    se non entrano è EXPORT_FAILED), estratti pin, stato presente, altri
    suggerimenti, limitazioni. Ritorna (testo, omissioni).
    """
    header_lines = [
        f"# Contesto di sessione: {session_id}",
        "",
        f"- Versione schema: {SCHEMA_VERSION}",
        f"- Backend: {backend}",
        "- Score: segnali di ranking, non percentuali di certezza",
    ]
    warnings_block = [MOCK_WARNING] if backend == "mock" else []
    mandatory_parts = ["\n".join(warnings_block), "\n".join(header_lines)]
    pin_refs: list[str] = ["## Ricordi pinnati"]
    for view in pinned:
        pin_refs.append(f"### {view.title}")
        pin_refs.append(f"- Testo completo: [{view.text_relpath}]({view.text_relpath})")
        pin_refs.append(f"- Fonte: {view.source_label}")
        pin_refs.append(f"- Selezione: {view.selection_label}")
    mandatory_parts.append("\n".join(pin_refs))
    mandatory = "\n\n".join(part for part in mandatory_parts if part)
    if len(mandatory) > budget_chars:
        raise ExportFailedError("EXPORT_FAILED: il contenuto obbligatorio non entra nel budget del contesto")

    used = len(mandatory)
    parts = [mandatory]
    omissions: list[str] = []

    def add_block(text: str, label: str) -> None:
        nonlocal used
        available = budget_chars - used - 2
        if available <= 0:
            omissions.append(f"{label}: omesso per budget")
            return
        fitted, truncated = _fit_paragraphs(text, available)
        if not fitted:
            omissions.append(f"{label}: omesso per budget")
            return
        parts.append(fitted)
        used += len(fitted) + 2
        if truncated:
            omissions.append(f"{label}: ridotto per budget, il resto resta negli allegati")

    clips = clip_transcripts or {}
    for view in pinned:
        note = (
            "\n\n_Estratto troncato: il testo completo è nel file linkato._"
            if len(view.extract) >= PIN_EXTRACT_CHARS
            else ""
        )
        clip_text = clips.get(view.memory_id)
        clip_block = f"\n\n**Trascrizione ASR:**\n\n{clip_text}" if clip_text is not None else ""
        add_block(
            f"**Estratto — {view.title}**\n\n{view.extract}{note}{clip_block}", f"estratto del pin {view.memory_id}"
        )
    mics = mic_transcripts or {}
    state_lines = ["## Stato recente"]
    for item in recent_inputs:
        if item.source == "mic_audio":
            line = f"- `mic_audio` ({item.duration_sec or 0:.1f}s alle {item.occurred_at:.0f})"
            if item.id in mics:
                line += f" — Trascrizione ASR: {mics[item.id]}"
            state_lines.append(line)
        else:
            state_lines.append(f"- `agent_text` {item.text}")
    add_block("\n".join(state_lines), "stato recente")
    suggestion_lines = ["## Altri suggerimenti"]
    if suggestions:
        for scored in suggestions:
            # Solo riferimento bibliografico: il file non è nel pacchetto, quindi
            # nessun link (un allegato indisponibile è voce di stato, non link rotto).
            suggestion_lines.append(
                f"- {scored.memory.context_header[:TITLE_MAX_CHARS]} · {scored.memory.modality}"
                " — solo riferimento, non incluso nel pacchetto"
            )
    else:
        suggestion_lines.append("_Nessun altro suggerimento._")
    add_block("\n".join(suggestion_lines), "altri suggerimenti")
    mode_notes = []
    if transcription_mode == "local":
        mode_notes.append("Le trascrizioni sono testo del modello ASR, non citazioni garantite.")
    else:
        mode_notes.append(
            "Modalità trascrizione off: gli allegati audio non sono trascritti — occorre un agente capace di ascoltare gli allegati."
        )
    limitation_lines = [
        "## Limitazioni e omissioni",
        *mode_notes,
        *(limitations or []),
        *(warnings or []),
        *OPERATIONAL_LIMITS,
        *omissions,
    ]
    add_block("\n".join(limitation_lines), "limitazioni")
    return "\n\n".join(parts), omissions


# --------------------------------------------------------------------------- trascrizione ASR


def _safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key) or "senza-nome"


def _audio_duration(path: Path) -> float | None:
    """Durata in secondi: wave per i WAV, ffprobe per gli altri media."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                return handle.getnframes() / rate if rate else None
        except (OSError, wave.Error):
            return None
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _transcription_cache_key(path: Path, revision: str, language: str, params_json: str) -> str:
    """Hash WAV + revisione ASR + lingua + parametri effettivi (deduplica export)."""
    digest = hashlib.sha256()
    with open(path, "rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(chr(0).join((revision, language, params_json)).encode("utf-8"))
    return digest.hexdigest()


def _cache_read(cache_dir: str | None, key: str) -> str | None:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{key}.txt"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _cache_write(cache_dir: str | None, key: str, text: str) -> None:
    if not cache_dir:
        return
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{key}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, directory / f"{key}.txt")


def _has_audio_stream(path: Path) -> bool | None:
    """True/False via ffprobe; None se non determinabile (si prova comunque l'ASR)."""
    if path.suffix.lower() == ".wav":
        return True
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def _run_transcription(
    settings: ExportSettings,
    staging: Path,
    selected: list[LiveInput],
    audio_assets: dict[str, str],
    pin_entries: list[dict],
    warnings: list[str],
    *,
    transcriber_factory=None,
    asr_cache_dir: str | None = None,
) -> tuple[dict, dict[str, str], dict[str, str]]:
    """Trascrive clip dei pin prima, poi parlato recente, entro il budget.

    Item-level all-or-nothing: mai troncamento silenzioso di una frase; gli
    ID oltre budget finiscono in excluded_ids con l'audio conservato. Ritorna
    (record_manifest, {input_id: testo} mic, {memory_id: testo} clip).
    """
    from src.hud import transcriber as asr

    record = {
        "mode": settings.transcription,
        "language": settings.language,
        "params": asdict(asr.AsrParams()),
        "budget_sec": settings.max_transcribe_sec,
        "used_sec": 0.0,
        "model": None,
        "repo": None,
        "revision": None,
        "per_input": [],
        "excluded_ids": [],
    }
    mic_transcripts: dict[str, str] = {}
    clip_transcripts: dict[str, str] = {}
    if settings.transcription != "local":
        return record, mic_transcripts, clip_transcripts
    targets: list[tuple[str, str, str]] = []
    for entry in pin_entries:
        for attachment in entry["attachments"]:
            if attachment["kind"] == "video_clip":
                targets.append((entry["memory_id"], "video_clip", attachment["path"]))
    for item in selected:
        rel = audio_assets.get(item.id)
        if rel:
            targets.append((item.id, "mic_audio", rel))
    ready, detail = (True, "factory di test")
    engine = None
    revision = ""
    if transcriber_factory is not None:
        # La factory incarna il motore (spy nei test): bypassa il preflight reale.
        engine = transcriber_factory(settings)
    else:
        ready, detail = asr.preflight_local(settings.asr_model_dir)
    if ready:
        model_manifest = asr.read_model_manifest(settings.asr_model_dir) or {}
        record["model"] = model_manifest.get("model")
        record["repo"] = model_manifest.get("repo")
        record["revision"] = revision = model_manifest.get("revision", "")
        if engine is None:
            engine = asr.LocalTranscriber(settings.asr_model_dir, settings.language)
    else:
        warnings.append(f"trascrizione local non pronta: {detail}; allegati audio conservati")
    params_json = json.dumps(record["params"], sort_keys=True)
    remaining = settings.max_transcribe_sec
    seen: dict[str, str] = {}
    stop = False
    for key, kind, rel in targets:
        path = staging / rel
        entry_record = {
            "key": key,
            "kind": kind,
            "audio": rel,
            "duration_sec": None,
            "status": "unavailable",
            "cache": "none",
            "transcript": None,
        }
        record["per_input"].append(entry_record)
        if not ready:
            continue
        # «Video senza audio resta clip visuale» (piano H2): niente ASR, nessun errore.
        if kind == "video_clip" and _has_audio_stream(path) is False:
            entry_record["status"] = "no_audio"
            continue
        duration = _audio_duration(path)
        entry_record["duration_sec"] = duration
        if stop:
            entry_record["status"] = "excluded"
            record["excluded_ids"].append(key)
            continue
        if duration is None:
            entry_record["status"] = "failed"
            warnings.append(f"trascrizione {key}: durata non determinabile, audio conservato")
            continue
        if duration > remaining + 1e-9:
            stop = True
            entry_record["status"] = "excluded"
            record["excluded_ids"].append(key)
            continue
        cache_key = _transcription_cache_key(path, revision, settings.language, params_json)
        text = seen.get(cache_key)
        if text is None:
            text = _cache_read(asr_cache_dir, cache_key)
        entry_record["cache"] = "hit" if text is not None else "miss"
        if text is None:
            try:
                text = engine.transcribe(str(path))
            except Exception as exc:
                entry_record["status"] = "failed"
                warnings.append(f"trascrizione {key} non riuscita: {exc}; audio conservato")
                continue
        text = text.strip()
        if not text:
            entry_record["status"] = "empty"
            warnings.append(f"trascrizione {key} vuota dal modello ASR; audio conservato")
            continue
        seen[cache_key] = text
        _cache_write(asr_cache_dir, cache_key, text)
        entry_record["status"] = "transcribed"
        transcript_rel = f"transcripts/{_safe_name(key)}.txt"
        (staging / "transcripts").mkdir(exist_ok=True)
        (staging / transcript_rel).write_text(text, encoding="utf-8")
        entry_record["transcript"] = transcript_rel
        remaining -= duration
        record["used_sec"] += duration
        if kind == "mic_audio":
            mic_transcripts[key] = text
        else:
            clip_transcripts[key] = text
    return record, mic_transcripts, clip_transcripts


# --------------------------------------------------------------------------- clipboard


def _copy_to_clipboard(markdown: str) -> tuple[str, list[str]]:
    """wl-copy se disponibile; assenza o errore non invalidano il pacchetto."""
    if shutil.which("wl-copy") is None:
        return "unavailable", []
    try:
        proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE, text=True)
        proc.communicate(markdown)
        return "copied", []
    except OSError as exc:
        return "failed", [f"clipboard non riuscito: {exc}"]


# --------------------------------------------------------------------------- verifica e pubblicazione


def _verify_package(staging: Path, manifest: dict) -> None:
    """Ogni riferimento esiste e ogni digest coincide, prima del rename."""
    for attachment in manifest["attachments"]:
        path = staging / attachment["path"]
        if not path.is_file():
            raise ExportFailedError(f"EXPORT_FAILED: allegato mancante {attachment['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != attachment["sha256"]:
            raise ExportFailedError(f"EXPORT_FAILED: digest errato per {attachment['path']}")
    for paths in manifest["memory_map"].values():
        for rel in paths:
            if not (staging / rel).is_file():
                raise ExportFailedError(f"EXPORT_FAILED: riferimento rotto {rel}")
    markdown = (staging / "context.md").read_text(encoding="utf-8")
    for target in set(re.findall(r"\]\(([^)]+)\)", markdown)):
        if not (staging / target).is_file():
            raise ExportFailedError(f"EXPORT_FAILED: link rotto nel contesto: {target}")


def assemble_and_export_context(
    state: LiveStateEvent,
    session_inputs: list[LiveInput],
    export_root: str,
    settings: ExportSettings,
    *,
    space: SpaceSpec | None = None,
    search_settings: SearchSettings | None = None,
    lease_expires_at: float | None = None,
    snapshot_provider=None,
    clock=time.time,
    transcriber_factory=None,
    asr_cache_dir: str | None = None,
) -> ExportResult:
    """Assembla e pubblica il pacchetto di contesto da uno snapshot coerente.

    Invalida `settings.context_max_chars` sotto il minimo (ValueError del
    chiamante). Se il lease scade durante l'acquisizione, chiude gli handle
    e riparte da un nuovo snapshot tramite `snapshot_provider`, senza
    pacchetto parziale. `transcriber_factory` e `asr_cache_dir` servono ai
    test (spy) e al worker H4. Errori fatali rimuovono lo staging e
    sollevano ExportFailedError('EXPORT_FAILED: ...').
    """
    if settings.context_max_chars < BUDGET_MIN_CHARS:
        raise ValueError(
            f"EXPORT_SETTINGS_INVALID: context_max_chars {settings.context_max_chars} sotto il minimo {BUDGET_MIN_CHARS}"
        )
    attempt_state, attempt_inputs, attempt_lease = state, session_inputs, lease_expires_at
    for attempt in range(MAX_LEASE_ATTEMPTS):
        try:
            return _assemble_once(
                attempt_state,
                attempt_inputs,
                export_root,
                settings,
                space=space,
                search_settings=search_settings,
                lease_expires_at=attempt_lease,
                clock=clock,
                transcriber_factory=transcriber_factory,
                asr_cache_dir=asr_cache_dir,
            )
        except _LeaseExpired:
            if snapshot_provider is None or attempt == MAX_LEASE_ATTEMPTS - 1:
                raise ExportFailedError(
                    "EXPORT_FAILED: lease scaduto durante l'acquisizione e nessun snapshot di ripetizione"
                ) from None
            attempt_state, attempt_inputs, attempt_lease = snapshot_provider()
    raise ExportFailedError("EXPORT_FAILED: lease scaduto ripetutamente")  # irraggiungibile


def _assemble_once(
    state: LiveStateEvent,
    session_inputs: list[LiveInput],
    export_root: str,
    settings: ExportSettings,
    *,
    space: SpaceSpec | None,
    search_settings: SearchSettings | None,
    lease_expires_at: float | None,
    clock,
    transcriber_factory=None,
    asr_cache_dir: str | None = None,
) -> ExportResult:
    """Un tentativo di assemblaggio: staging nuovo, pubblicazione con rename atomico."""
    export_id = uuid.uuid4().hex
    session_dir = Path(export_root) / state.session_id
    staging = session_dir / f".staging-{export_id}"
    final = session_dir / export_id
    (staging / "assets").mkdir(parents=True)
    (staging / "memories").mkdir()
    try:
        selected, missing = _select_inputs(state, session_inputs)
        warnings = [f"input {input_id} dello snapshot non trovato nella storia" for input_id in missing]
        audio_assets, audio_warnings = _acquire_audio(
            selected, staging, lease_expires_at=lease_expires_at, clock=clock
        )
        warnings.extend(audio_warnings)
        views, pin_entries, memory_map, media_attachments, pin_warnings, limitations = _export_pinned(state, staging)
        warnings.extend(pin_warnings)
        transcription, mic_transcripts, clip_transcripts = _run_transcription(
            settings,
            staging,
            selected,
            audio_assets,
            pin_entries,
            warnings,
            transcriber_factory=transcriber_factory,
            asr_cache_dir=asr_cache_dir,
        )
        pinned_ids = {pin.memory.id for pin in state.pinned}
        suggestions = [scored for scored in state.results if scored.memory.id not in pinned_ids][
            :MAX_OTHER_SUGGESTIONS
        ]
        suggestions_record = [
            {
                "memory_id": scored.memory.id,
                "title": scored.memory.context_header[:TITLE_MAX_CHARS],
                "source_path": scored.memory.file_path,
                "modality": scored.memory.modality,
            }
            for scored in suggestions
        ]
        markdown, omissions = _build_markdown(
            backend=state.backend,
            session_id=state.session_id,
            pinned=views,
            recent_inputs=selected,
            suggestions=suggestions,
            budget_chars=settings.context_max_chars,
            warnings=warnings,
            limitations=limitations,
            mic_transcripts=mic_transcripts,
            clip_transcripts=clip_transcripts,
            transcription_mode=settings.transcription,
        )
        (staging / "context.md").write_text(markdown, encoding="utf-8")

        audio_by_id = {item.id: item for item in selected}
        attachments = [
            {
                "path": rel,
                "sha256": hashlib.sha256((staging / rel).read_bytes()).hexdigest(),
                "kind": "mic_audio",
                "source_path": audio_by_id[input_id].audio_path,
            }
            for input_id, rel in audio_assets.items()
        ]
        attachments.extend(media_attachments)
        clipboard_status, clipboard_warnings = _copy_to_clipboard(markdown)
        warnings.extend(clipboard_warnings)
        mic_status = {
            item["key"]: item["status"] for item in transcription["per_input"] if item["kind"] == "mic_audio"
        }
        session_context = {
            "session_id": state.session_id,
            "snapshot": {"id": state.event_id, "emitted_at": state.emitted_at},
            "inputs": [
                {
                    "id": item.id,
                    "source": item.source,
                    "occurred_at": item.occurred_at,
                    "duration_sec": item.duration_sec,
                    "text": item.text if item.source == "agent_text" else "",
                    "audio_path": audio_assets.get(item.id),
                    "transcription": mic_transcripts.get(item.id),
                    "transcription_status": mic_status.get(item.id),
                }
                for item in selected
            ],
        }
        (staging / "session-context.json").write_text(
            json.dumps(session_context, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "export_id": export_id,
            "session_id": state.session_id,
            "space_id": state.space_id,
            "backend": state.backend,
            "space_spec": asdict(space) if space else None,
            "search_settings": asdict(search_settings) if search_settings else None,
            "export_settings": asdict(settings),
            "created_at": clock(),
            "snapshot": {
                "event_id": state.event_id,
                "sequence": state.sequence,
                "emitted_at": state.emitted_at,
                "reason": state.reason,
            },
            "inputs": session_context["inputs"],
            "pinned": pin_entries,
            "suggestions": suggestions_record,
            "memory_map": memory_map,
            "attachments": attachments,
            "transcription": transcription,
            "clipboard_status": clipboard_status,
            "warnings": warnings,
            "omissions": omissions,
            "operational_limits": OPERATIONAL_LIMITS,
        }
        _verify_package(staging, manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.rename(staging, final)
        return ExportResult(
            export_dir=str(final),
            markdown_path=str(final / "context.md"),
            manifest_path=str(final / "manifest.json"),
            clipboard_status=clipboard_status,
            warnings=warnings,
        )
    except _LeaseExpired:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except ExportFailedError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ExportFailedError(f"EXPORT_FAILED: {exc}") from exc
