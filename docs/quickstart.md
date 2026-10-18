# Quickstart — Ambient Second Brain (PoC)

Stato: verifica finale in corso (2026-10-18). Scope: text-only per decisione
utente; S2-audio FAIL reale documentato in `docs/evaluation/probe-failed-run{1,2}.json`.
Comandi verificati dove indicato; gli altri vengono consolidati al termine
della verifica e2e.

## Prerequisiti

- Fedora 44 / GNOME 50.5 / Wayland (vedi `docs/environment.md`).
- `uv sync` installa lo stack (torch cu130, transformers, GTK4/Adw, ffmpeg via sistema).
- Requisiti di sistema verificati da: `uv run python -m src.preflight --backend real`
  (exit 0 = pronto; exit 2 stampa le correzioni proposte).
- **Gap noto**: `mpv` non installato — per aprire i video al secondo esatto nel
  percorso HUD: `sudo dnf install -y mpv` (non installabile dall'ambiente di
  orchestrazione; non blocca l'headless).

## Build fixture di valutazione (una volta)

```bash
uv run python scripts/build_fixtures.py --output tests/fixtures/ambient
# corpus 16 file neutri + 22 query WAV + manifest, riproducibile bit-exact
```

## Modalità headless (verificata in smoke)

```bash
# smoke automatico: avvio reale, snapshot, suggerimento live, arresto pulito
bash scripts/smoke_headless.sh
```

Avvio manuale equivalente:

```bash
LOG=./logs/agent-session.log; : > $LOG
uv run python run_poc.py --watch-dir ./data/watch-corpus --backend real \
    --headless --no-mic --agent-log $LOG --agent-format plain
# in un altro terminale, con <run_dir> da $XDG_RUNTIME_DIR/artificial-memory/:
uv run python -m src.control --runtime <run_dir> snapshot
uv run python -m src.control --runtime <run_dir> pin <MEMORY_ID>
uv run python -m src.control --runtime <run_dir> mic on|off   # solo diagnostica (fuori gate)
```

Un nuovo messaggio nel log bridge (`script -q -f -c 'claude' ./logs/agent-session.log`)
produce suggerimenti entro ~1 s (p95 misurato 1.047 s).

## Modalità desktop (HUD)

```bash
uv run python run_poc.py --watch-dir ./data/watch-dir --backend real \
    --agent-log ./logs/agent-session.log --agent-format plain
# trascrizione default off; per l'export local:
#   uv sync --extra asr
#   uv run python -m src.hud.transcriber prepare --model base --output ./data/models/asr
#   ... --transcription local --asr-model-dir ./data/models/asr
```

## Valutazione

```bash
uv run python scripts/evaluate.py --split calibration --backend real --report docs/evaluation/calibration.json
uv run python scripts/evaluate.py --split test --backend real --report docs/evaluation/final.json
```

## Permessi e persistenza

- Directory runtime in `$XDG_RUNTIME_DIR/artificial-memory/<run_id>/` (0700),
  rimossa dall'arresto; DB `data/brain-{real,mock}.db`, grafo incluso; export
  in `exports/`, durevoli.
- Il DB è protetto da lock di processo: una seconda istanza sullo stesso DB
  esce con errore esplicito.
- Reset sessione (`new-session`) e arresto eliminano lo storico temporaneo e
  i WAV live; pin/grafo/export restano.
