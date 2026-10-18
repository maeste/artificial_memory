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
| Agent-1-Storage | ST1–ST4 | ✅ CHIUSO (nessun residuo) | import dimostrativo bank reale: 26 record/16 file, generation 16, cos 1.0, mismatch rifiutato — su DB usa-e-getta |
| Agent-2-Inference | I1–I4 | ✅ CHIUSO (nessun residuo) | Fase 1: text PASS / audio FAIL documentato; revisione congelata; bank consegnato e importato; disponibile per supporto |
| Agent-3-Ingestion | G1–G4 | 🟢 57 test verificati (fix TIMEOUT + stack reale mock) | VIA al run brain-real.db trattenuto fino a fine re-probe I4 (isolamento misure) |
| Agent-4-Sensors | L1–L4 | ✅ TASK_DONE accettato | 94 test + ruff verificati da me; committato; ReplayHarness allineata a evaluate.py; verifiche congiunte Fase 4 dopo gate reale |
| Agent-5-HUD | H1–H4 | 🟢 H1+H2+H3 ✅ verificate | H3 chiuso: 9 transcription + 81 ordinari verificati da me; smoke pipeline reale (termini sopravvissuti C2/C3) + limite ASR su TTS dichiarato in README; atteso TASK_DONE formale H4 |

## O3–O5 — Launcher, gate, verifica finale

| Task | Stato | Evidenza |
|---|---|---|
| O3 test launcher con figli fake | ✅ | `tests/e2e/test_launcher.py`: timeout startup, morte figlio indispensabile, SIGINT+teardown inverso senza orfani, lock DB, SPACE_MISMATCH (5 test) |
| O3 CLI `run_poc.py` | ✅ | flag richiesti (watch-dir/backend/agent-log/agent-format/no-mic/device/db/show-all/headless/calibration/transcription/asr-model-dir); revisione reale richiesta o da data/probe-config.json |
| O3 CLI `python -m src.control` | ✅ | snapshot/pin/unpin/new-session/mic on\|off con retry stesso command_id |
| O3 quickstart | 🟢 bozza | `docs/quickstart.md`; comandi consolidati a consegna |
| O3/O5 smoke headless reale | 🟡 bloccato su serve | launcher corretto (`-m src.inference serve`); watcher e runner verificati attivi col config integrale (preemptive check); `serve` rifiuta i campi degli altri figli → FIX_REQUEST #2 ad Agent-2 (contratto §6: config passato integralmente) |
| O4 gate incrementali | 🟢 avanzato | Fase 1: testo PASS / audio FAIL documentato (run-1+2). Calibrazione congelata cosine/0.70/0.70 (griglia estesa a 0.70, direzione più severa, criteri invariati). **Gate finale text: 8/8 hit rank 1 (S1/S2/S3 PASS), astensione 5/5 (S4 PASS), S5 PASS (p50 1.038 s / p95 1.047 s su 39 campioni, debounce incluso)**. Checklist manuale creata in docs/evaluation/manual.md. |
| O5 verifica finale | non avviato | — |

Commit checkpoint: `b8aeb95` (solo file di pertinenza Orchestrator; fixture
corpus rigenerabili bit-exact, escluse via .gitignore, checksum nel manifest).

## Registro blocker/decisioni

- 2026-10-18 **GATE FASE 1 PRELIMINARE: FAIL REALE su S2-audio**. Run-1 (voce
  unica, confound TTS) e run-2 (fixture v1.1, confound rimosso) entrambi audio
  2/4 con markdown/pdf mancanti; testo 4/4 rank 1 su tutti i formati, p95 41 ms.
  Evidenze: `docs/evaluation/probe-failed-run1.json`, `probe-failed-run2.json`,
  `manifest-run1.json`; test real_model del gate rosso come prova. Nessun
  cambiamento ulteriore di fixture/query/gold/soglie/modello (D4). Decisione
  obiettivi (text-only vs diagnostica voce reale) in escalation all'utente
  come da PRD §11. Modello dichiarato libero: Agent-3 autorizzato al run
  reale su brain-real.db.
- 2026-10-18 **[BLOCKER] gate Fase 1 run-1 (Agent-2)**: testo 4/4, audio 2/4
  (C1/C2 mancate; cluster vocale piatto 0.66–0.73). Diagnosi accettata: confound
  voce-query == voce-narrazione (stesso TTS espeak-ng) → matching audio-audio
  maschera audio→testo. **Decisione Orchestrator**: fix fixture v1.1 (query
  `it+f4`, narrazioni `it`); 16 file corpus byte-identici (checksum 826931ee…
  invariato), 22 WAV query rigenerati, anomalie documentate nel manifest
  PRIMA della nuova prova; manifest run-1 conservato in
  `docs/evaluation/manifest-run1.json`; report fallito atteso in
  `docs/evaluation/probe-failed-run1.json`. Se l'audio resta <4/4 senza
  confound → FAIL REALE S2-audio, evidenza conservata, decisione all'utente.
- 2026-10-18 **SpaceSpec reale congelato**: `google/embeddinggemma-2` @ revisione
  `914f7f89142e33e77833254d9c9b90c3cef7303b`, space_id
  `a909a1662bdfb650f3db8845b3171986e1aa41b84e97984709094dc2085f7b13`,
  scritto in `data/probe-config.json`. Dipendenza nuova obbligatoria `torchvision`
  (processor video caricato a import) aggiunta a pyproject e rilockata; build
  locked riverificata con test `real_model` (2 passed, CUDA attivo).
- 2026-10-18 API sbloccate per Agent-3/Agent-4 dopo verifica: EmbedClient +
  SemanticHashMock (I1/I2), BrainStorage CRUD (ST1); replace_file in arrivo con
  ST2, hybrid_search con ST3.
- 2026-10-18 Orchestrator: nessun blocker tecnico O1. Revisione reale del modello
  (commit HF di `google/embeddinggemma-2`) da risolvere con Agent-2 prima del
  congelamento SpaceSpec; l'identità dello spazio resta fissata all'avvio.
- 2026-10-18 Orchestrator: Agent-5-HUD riavviato → nuovo peer ID
  `e4d5cf126c244a54bc7f5a1b1a035ab2`, conversazione di presenza riaperta (`1fbfa5f3`)
  con recap; sollecito protocollo (risposta dovuta prima del [START]) inviato a 1/3/4.
