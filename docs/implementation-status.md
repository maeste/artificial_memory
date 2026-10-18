# Stato di implementazione — Ambient Second Brain

Aggiornato dall'Orchestrator: 2026-10-18. Versione contratto: **1.0**.
Colonne: task, stato, evidenza, blocker. `PASS reale` / `FAIL reale` / `mock-only`
/ `manuale non eseguito` restano distinti in ogni gate.

## Presenza esecutori (O2)

| Esecutore | Peer ID Lince | Conv. | Presenza | [START] |
|---|---|---|---|---|
| Agent-1-Storage | `dcd9a03357794b1b8a29d02738c8dc90` | `91460ed4` | ✅ | ✅ inviato (ST1→ST4) |
| Agent-2-Inference | `4aa82691ea6b4b36a0a4f95d93420507` | `38990c88` | ✅ | ✅ inviato (**priorità** client+P1 reale) |
| Agent-3-Ingestion | `b22e3d3c693e4b5f91141006c9a9b1ce` | `747fcfca` | ✅ | ✅ inviato (G1→G4, fake fino a INTERFACE_READY 1/2) |
| Agent-4-Sensors | `be0508552679420fb259f19d0597e845` | `e907923e` | ✅ | ✅ inviato (L1→L4, nota interfaccia ReplayHarness) |
| Agent-5-HUD | `e4d5cf126c244a54bc7f5a1b1a035ab2` | `1fbfa5f3` | ✅ | ✅ inviato (H1→H4, mpv gap segnalato) |

Dispatch completato 2026-10-18: contratto 1.0 (src/contracts.py), albero b8aeb95,
dipendenze venv e corpus fixture comunicati a ciascuno. Tag attesi:
[INTERFACE_READY] [BLOCKER] [CONTRACT_CHANGE] [TASK_DONE].

## O1 — Ambiente, contratti e fixture (Orchestrator)

| Task | Stato | Evidenza |
|---|---|---|
| Rilevamento ambiente → `docs/environment.md` | ✅ | Fedora 44, GNOME 50.5, Wayland, PipeWire 1.6.9, i9-13950HX, 62 GiB RAM, RTX 5000 Ada (CUDA), 77 G liberi |
| Cinque package + data/export/runtime | ✅ | `src/{storage,inference,ingestion,sensors,hud}/`, `data/`, `exports/` |
| `src/contracts.py` con decoder ricorsivi (TDD) | ✅ | RED verificato su modulo assente → GREEN: `uv run pytest tests/contracts -q` = 54 passed |
| Roundtrip eventi annidati/pin/snapshot, invalidi, NaN/Inf, norme | ✅ | `tests/contracts/test_wire.py` (54 test) |
| `scripts/build_fixtures.py` + corpus + manifest | ✅ | 16 file, 22 query, 26 document_requests; corpus_checksum `826931ee…` riproducibile bit-exact su due rigenerazioni |
| `scripts/evaluate.py` (metriche e griglia 08) | ✅ skeleton | metriche/calibrazione testate (8 test); esecuzione end-to-end richiede `src.sensors.replay` (Agent-4); il gate finale richiede backend real |
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
| Agent-1-Storage | ST1–ST4 | 🟢 avviato | START consegnato; attesi artefatti |
| Agent-2-Inference | I1–I4 (priorità client + prova reale P1) | 🟢 avviato | START consegnato; attesa revisione HF risolta |
| Agent-3-Ingestion | G1–G4 | 🟢 avviato | fake locali fino a INTERFACE_READY 1/2 |
| Agent-4-Sensors | L1–L4 | 🟢 avviato | possibile CONTRACT_CHANGE su ReplayHarness |
| Agent-5-HUD | H1–H4 | 🟢 avviato | mpv assente (gate manuale futuro) |

## O3–O5 — Launcher, gate, verifica finale

| Task | Stato | Evidenza |
|---|---|---|
| O3 test launcher con figli fake | ✅ | `tests/e2e/test_launcher.py`: timeout startup, morte figlio indispensabile, SIGINT+teardown inverso senza orfani, lock DB, SPACE_MISMATCH (5 test) |
| O3 CLI `run_poc.py` | ✅ | flag richiesti (watch-dir/backend/agent-log/agent-format/no-mic/device/db/show-all/headless/calibration/transcription/asr-model-dir); revisione reale richiesta o da data/probe-config.json |
| O3 CLI `python -m src.control` | ✅ | snapshot/pin/unpin/new-session/mic on\|off con retry stesso command_id |
| O3 quickstart | ⏳ | scritto alla consegna con i comandi verificati end-to-end |
| O4 gate incrementali | non avviato | dipende dagli artefatti degli esecutori |
| O5 verifica finale | non avviato | — |

Commit checkpoint: `b8aeb95` (solo file di pertinenza Orchestrator; fixture
corpus rigenerabili bit-exact, escluse via .gitignore, checksum nel manifest).

## Registro blocker/decisioni

- 2026-10-18 Orchestrator: nessun blocker tecnico O1. Revisione reale del modello
  (commit HF di `google/embeddinggemma-2`) da risolvere con Agent-2 prima del
  congelamento SpaceSpec; l'identità dello spazio resta fissata all'avvio.
- 2026-10-18 Orchestrator: Agent-5-HUD riavviato → nuovo peer ID
  `e4d5cf126c244a54bc7f5a1b1a035ab2`, conversazione di presenza riaperta (`1fbfa5f3`)
  con recap; sollecito protocollo (risposta dovuta prima del [START]) inviato a 1/3/4.
