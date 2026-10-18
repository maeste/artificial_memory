# Ambiente di esecuzione — rilevamento Orchestrator (O1)

Data rilevamento: 2026-10-18. Comandi essenziali eseguiti sulla macchina bersaglio.
Il preflight automatico (`src/preflight.py`) ripete questi controlli a ogni avvio.

## Sistema

| Voce | Valore rilevato | Stato |
|---|---|---|
| OS | Fedora release 44 (Forty Four) | OK |
| Desktop | GNOME Shell 50.5 | OK |
| Sessione | Wayland (`XDG_SESSION_TYPE=wayland`, `wayland-0`) | OK |
| Audio | PipeWire 1.6.9 compilato/linked (libpipewire 1.6.9) | OK |
| Python sistema | 3.14.8 (`/usr/bin/python3`) | Vedasi nota versione |
| uv | 0.11.19 | OK |

## Hardware

| Voce | Valore |
|---|---|
| CPU | 13th Gen Intel Core i9-13950HX, 24C/32T |
| RAM | 62 GiB totali |
| GPU | NVIDIA RTX 5000 Ada Generation Laptop GPU (AD103GLM) + Intel UHD Raptor Lake-S (iGPU) |
| Disco `/home` | 953G totali, **77G liberi (92% usato)** al rilevamento |

Nota disco: i download dei pesi (modello embedding + eventuale ASR base) e lo staging
export vanno contenuti nei ~77G liberi; registrato come limite operativo del PoC.

## Strumenti di sistema

| Strumento | Versione/posizione | Stato |
|---|---|---|
| ffmpeg / ffprobe | 8.1.3 | OK (decode fixture MP4 da verificare in preflight) |
| evince (viewer PDF, `--page-index`) | presente (`/usr/bin/evince`) | OK, salto pagina da verificare in gate manuale |
| xdg-open | presente | OK |
| wl-copy / wl-paste | presenti | OK (clipboard Wayland) |
| espeak-ng | 1.52.0 | OK (voci it per query WAV sintetiche) |
| parec | presente (`/usr/bin/parec`) | OK (fallback cattura mic) |
| **mpv** | **ASSENTE** | **GAP**: richiesto da HUD per apertura MP4 al secondo esatto. Correzione proposta: `sudo dnf install -y mpv`. Il preflight lo segnala; non installato automaticamente. |
| GTK4 / Libadwaita typelibs | `Gtk-4.0.typelib`, `Adw-1.typelib` in `/usr/lib64/girepository-1.0` | OK (binding PyGObject da installare via uv) |

## Python del progetto

Il piano richiede Python 3.12+. Il Python di sistema è 3.14.8, per il quale la
disponibilità di wheel binarie di PyTorch/transformers non è garantita; il progetto
usa quindi un interprete gestito da uv pinnato a 3.12 (`.python-version`), coerente
con lo stack dichiarato. La versione effettiva viene registrata nel report Fase 1.

## Dipendenze Python (da bloccare in `uv.lock` dopo smoke test)

Base: `numpy`, `pyzmq`, `sqlite-vec`, `watchdog`, `pymupdf`, `pillow`, `sounddevice`,
`soundfile`, `torch`, `transformers`, `sentence-transformers`, `torchcodec`, `PyGObject`,
`pycairo`, `pytest`.
Extra opzionale `asr`: `faster-whisper` (`uv sync --extra asr`), richiesto solo con
`--transcription local` (default `off`: nessun download ASR).
