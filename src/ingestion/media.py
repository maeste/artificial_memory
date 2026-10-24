"""Operazioni media (G2): render pagine PDF, decode immagini, probe/estrazione MP4.

Tutti i comandi esterni (ffmpeg/ffprobe) girano senza shell, con lista di
argomenti e timeout esplicito; i prodotti finiscono nella staging del file e
restano validi fino al termine dell'indicizzazione (li chiude index_file).
Gli errori sono MediaDecodeError: il dispatch degli estrattori li traduce in
InvalidMediaError senza perdere il dettaglio.
"""

import json
import subprocess
from pathlib import Path

FFMPEG_TIMEOUT_SEC = 120
PROBE_TIMEOUT_SEC = 60
RENDER_DPI = 120
MAX_RENDER_SIDE_PX = 1600


class MediaDecodeError(Exception):
    """Decode o estrazione media fallita (ffmpeg/ffprobe/Pillow/pymupdf)."""


def _run(cmd: list[str], timeout: int) -> None:
    """Subprocess senza shell: lista argomenti, output catturato, timeout."""
    target = Path(cmd[-1]).name
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MediaDecodeError(f"Timeout di {cmd[0]} su {target}") from exc
    except OSError as exc:
        raise MediaDecodeError(f"{cmd[0]} non avviabile: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip().splitlines()
        detail = stderr[-1] if stderr else f"rc={result.returncode}"
        raise MediaDecodeError(f"{cmd[0]} fallito su {target}: {detail}")


# --------------------------------------------------------------------------- PDF


def render_page_png(page, staging: Path, page_num: int) -> Path:
    """Render della pagina a 120 DPI con lato massimo 1600 px, in staging."""
    import pymupdf

    staging.mkdir(parents=True, exist_ok=True)
    rect = page.rect
    long_side_px = max(rect.width, rect.height, 1.0) / 72.0 * RENDER_DPI
    zoom = RENDER_DPI / 72.0
    if long_side_px > MAX_RENDER_SIDE_PX:
        zoom *= MAX_RENDER_SIDE_PX / long_side_px
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    out = staging / f"page{page_num}.png"
    pix.save(str(out))
    return out


# --------------------------------------------------------------------------- immagini


def decode_image(path: Path) -> tuple[int, int, str]:
    """Valida il decode Pillow e restituisce (larghezza, altezza, modalità)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            return image.width, image.height, image.mode
    except Exception as exc:
        raise MediaDecodeError(f"Immagine non decodificabile: {path.name}: {exc}") from exc


def save_rgb_copy(path: Path, staging: Path) -> Path:
    """Copia RGB in staging: l'alfa viene appiattita su bianco, non su nero."""
    from PIL import Image

    staging.mkdir(parents=True, exist_ok=True)
    out = staging / f"{path.stem}_rgb.png"
    try:
        with Image.open(path) as image:
            if image.mode in ("RGBA", "LA", "PA"):
                flattened = image.convert("RGBA")
                base = Image.new("RGB", flattened.size, (255, 255, 255))
                base.paste(flattened, mask=flattened.split()[3])
                base.save(str(out))
            else:
                image.convert("RGB").save(str(out))
    except Exception as exc:
        raise MediaDecodeError(f"Conversione RGB fallita: {path.name}: {exc}") from exc
    return out


# --------------------------------------------------------------------------- MP4


def probe_video(path: Path) -> dict:
    """ffprobe JSON con format e stream; probe vuota o corrotta è errore."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        raise MediaDecodeError(f"Timeout di ffprobe su {path.name}") from exc
    except OSError as exc:
        raise MediaDecodeError(f"ffprobe non avviabile: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip().splitlines()
        detail = stderr[-1] if stderr else f"rc={result.returncode}"
        raise MediaDecodeError(f"ffprobe fallito su {path.name}: {detail}")
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise MediaDecodeError(f"Probe non interpretabile per {path.name}") from exc
    if not isinstance(data, dict) or not data.get("format"):
        raise MediaDecodeError(f"Probe senza formato per {path.name}")
    return data


def video_duration(probe: dict) -> float:
    """Durata della traccia video: le finestre riguardano i frame, non il
    contenitore (il padding audio AAC allunga il container di decimi)."""
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        value = stream.get("duration")
        if value is None:
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    try:
        return float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaDecodeError("Durata video non disponibile dalla probe") from exc


def has_audio_stream(probe: dict) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def extract_frame(path: Path, at_sec: float, out: Path) -> Path:
    """Frame singolo al secondo richiesto; frame assente/vuoto è errore."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{at_sec:.3f}", "-i", str(path), "-frames:v", "1", str(out)],
        FFMPEG_TIMEOUT_SEC,
    )
    if not out.exists() or out.stat().st_size == 0:
        raise MediaDecodeError(f"Frame assente a {at_sec:.3f}s in {path.name}")
    return out


def extract_audio_window(path: Path, start_sec: float, end_sec: float, out: Path) -> Path:
    """WAV mono PCM 16 kHz della finestra [start, end)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{end_sec - start_sec:.3f}",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-vn",
            str(out),
        ],
        FFMPEG_TIMEOUT_SEC,
    )
    return out
