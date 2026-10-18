# Stato di implementazione — Ambient Second Brain

Aggiornato dall'Orchestrator: 2026-10-18. Versione contratto: **1.0**.
Colonne: task, stato, evidenza, blocker. `PASS reale` / `FAIL reale` / `mock-only`
/ `manuale non eseguito` restano distinti in ogni gate.

## Presenza esecutori (O2)

| Esecutore | Peer ID Lince | Conv. | Presenza |
|---|---|---|---|
| Agent-1-Storage | `dcd9a03357794b1b8a29d02738c8dc90` | `91460ed4` | ⏳ in attesa |
| Agent-2-Inference | `4aa82691ea6b4b36a0a4f95d93420507` | `38990c88` | ✅ [PRESENT] |
| Agent-3-Ingestion | `b22e3d3c693e4b5f91141006c9a9b1ce` | `747fcfca` | ⏳ in attesa |
| Agent-4-Sensors | `be0508552679420fb259f19d0597e845` | `e907923e` | ⏳ in attesa |
| Agent-5-HUD | `1cbd2f8646844399a0baf0368c3062a0` | `c581f666` | ⏳ in attesa |

Il dispatch [START] parte solo quando tutti e cinque hanno risposto [PRESENT].

## O1 — Ambiente, contratti e fixture (Orchestrator)

| Task | Stato | Evidenza |
|---|---|---|
| Rilevamento ambiente → `docs/environment.md` | ✅ | Fedora 44, GNOME 50.5, Wayland, PipeWire 1.6.9, i9-13950HX, 62 GiB RAM, RTX 5000 Ada (CUDA), 77 G liberi |
| Cinque package + data/export/runtime | ✅ | `src/{storage,inference,ingestion,sensors,hud}/`, `data/`, `exports/` |
| `src/contracts.py` con decoder ricorsivi (TDD) | ✅ | RED verificato su modulo assente → GREEN: `uv run pytest tests/contracts -q` = 54 passed |
| Roundtrip eventi annidati/pin/snapshot, invalidi, NaN/Inf, norme | ✅ | `tests/contracts/test_wire.py` (54 test) |
| `scripts/build_fixtures.py` + corpus + manifest | ✅ | 16 file, 22 query, 26 document_requests; corpus_checksum `826931ee…` riproducibile bit-exact su due rigenerazioni |
| `scripts/evaluate.py` (metriche e griglia 08) | ✅ skeleton | metriche/calibrazione testate (8 test); esecuzione end-to-end richiede `src.sensors.replay` (Agent-4) |
| Marker pytest real_model/desktop/hardware | ✅ | `pyproject.toml` |
| Dipendenze `uv.lock` | ⚠️ bloccato a smoke test | `uv sync` OK: torch 2.14.1+cu130 (CUDA attivo), transformers 5.20.1; convalida finale dopo smoke reale Agent-2 |
| Preflight `--backend real` | ✅ | `ready: true`, obbligatori tutti OK; warn `mpv` assente |

### Ambiente — gap aperti

| Gap | Impatto | Correzione proposta |
|---|---|---|
| `mpv` assente | Gate Fase 5 (apertura MP4 al secondo) | `sudo dnf install -y mpv` (da eseguire prima del gate HUD) |

## Avanzamento esecutori

| Esecutore | Task assegnati | Stato | Evidenza / blocker |
|---|---|---|---|
| Agent-1-Storage | ST1–ST4 | non avviato | attesa [START] |
| Agent-2-Inference | I1–I4 (priorità client + prova reale P1) | non avviato | attesa [START] |
| Agent-3-Ingestion | G1–G4 | non avviato | attesa [START] |
| Agent-4-Sensors | L1–L4 | non avviato | attesa [START] |
| Agent-5-HUD | H1–H4 | non avviato | attesa [START] |

## O3–O5 — Launcher, gate, verifica finale

| Task | Stato |
|---|---|
| O3 launcher + CLI controllo | non avviato (lavoro indipendente possibile in attesa delle presenze) |
| O4 gate incrementali | non avviato |
| O5 verifica finale | non avviato |

## Registro blocker/decisioni

- 2026-10-18 Orchestrator: nessun blocker tecnico O1. Revisione reale del modello
  (commit HF di `google/embeddinggemma-2`) da risolvere con Agent-2 prima del
  congelamento SpaceSpec; l'identità dello spazio resta fissata all'avvio.
