"""Comandi di apertura senza shell (piano HUD H1).

Gli argv sono vettori di argomenti: mai shell=True, mai interpretazione di
URI/commandi dal contenuto dei documenti. mpv ed evince permettono di
raggiungere il punto esatto del ricordo; xdg-open copre i file generici.
"""

import os
import shutil
import subprocess

from src.contracts import MemoryView

VIDEO_PLAYER = "mpv"
PDF_VIEWER = "evince"
GENERIC_OPENER = "xdg-open"


def _format_seconds(start_sec: float) -> str:
    """Formatta i secondi conservando i decimali (12.5 resta '12.5')."""
    return f"{float(start_sec)}"


def build_open_command(memory: MemoryView) -> list[str]:
    """Vettore argv per aprire il ricordo nel punto selezionato.

    MP4 `mpv --start=<start_sec> file`; PDF `evince --page-index=<page_num>
    file` con numero pagina dell'utente (1-based, nonostante il nome dell'
    opzione); altri `xdg-open file`. Path con spazi restano un solo argomento.
    """
    if memory.modality == "video":
        argv = [VIDEO_PLAYER]
        if memory.start_sec is not None:
            argv.append(f"--start={_format_seconds(memory.start_sec)}")
        argv.append(memory.file_path)
        return argv
    if memory.modality == "pdf":
        argv = [PDF_VIEWER]
        if memory.page_num is not None:
            argv.append(f"--page-index={memory.page_num}")
        argv.append(memory.file_path)
        return argv
    return [GENERIC_OPENER, memory.file_path]


def missing_open_tools() -> list[str]:
    """Preflight: nomi degli strumenti di apertura assenti sul sistema."""
    required = (VIDEO_PLAYER, PDF_VIEWER, GENERIC_OPENER)
    return [name for name in required if shutil.which(name) is None]


def open_memory_file(memory: MemoryView) -> None:
    """Apre il file del ricordo con `subprocess.Popen(argv)`, senza shell.

    Verifica prima l'esistenza del file (errore visibile se assente) e la
    disponibilità dello strumento richiesto.
    """
    if not os.path.isfile(memory.file_path):
        raise FileNotFoundError(f"FILE_MISSING: il file del ricordo non esiste: {memory.file_path}")
    argv = build_open_command(memory)
    if shutil.which(argv[0]) is None:
        raise RuntimeError(f"TOOL_MISSING: {argv[0]} non è disponibile per aprire {memory.file_path}")
    subprocess.Popen(argv)
