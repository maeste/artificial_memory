# Stato di implementazione — Ambient Second Brain

Aggiornato dall'Orchestrator: 2026-10-18, verifica finale. Versione contratto: **1.0**
(+ note di revisione datate in 07 §5 e 08 §3). Scope: **text-only per decisione
utente**; S2-audio FAIL reale documentato. Distinzioni mantenute ovunque:
`PASS reale` / `FAIL reale` / `mock-only` / `manuale non eseguito`.

## Criteri globali S1–S8

| ID | Esito | Prova |
|---|---|---|
| S1 | ✅ PASS reale (text) | gate finale `docs/evaluation/final.json`: 8/8 rank 1; smoke live e2e (f01.md top da messaggio agente reale) |
| S2 | ✅ testo 8/8 ≥ 7/8 · ✅ **audio PASS con voce reale (run-3, 4/4)** | `final.json`; `probe-failed-run{1,2}.json` (TTS FAIL, conservati); `probe-real-voice-run3.json` (voce reale 4/4, tutti i formati, decisione utente opzione A, nota datata in 08 §3) |
| S3 | ✅ PASS reale (text) | `final.json` metrics_per_format: markdown/pdf/image/video tutte 1.0 |
| S4 | ✅ PASS reale (text) | `final.json`: astensione 5/5 negative test |
| S5 | ✅ PASS reale (text) | `final.json` timings: p50 1.038 s / p95 1.047 s (39 campioni, debounce incluso; percorso mic non misurato) |
| S6 | ✅ PASS deterministico | `src/storage/test_ranking.py`: A=1.6/B=1.3/C=1.4, delta .25 γ.6 → B 1.54 > C; embedding invariati |
| S7 | ✅ PASS reale | export off: `docs/evaluation/sample-export/`; export local: `docs/evaluation/sample-export-local/` (ASR reale, limiti dichiarati) |
| S8 | ✅ PASS (unit/integrazione) | `src/ingestion/test_watcher.py` lifecycle scan→modifica→delete→restart→move; `src/storage` cascata/archi; riavvio zero re-embedding (smoke reale 477→unchanged e 16→unchanged) |

## Esecuzione O5 — prove e comandi

| Prova | Esito | Evidenza |
|---|---|---|
| `uv run pytest -m 'not real_model and not desktop and not hardware' -q` | ✅ 410 passed, 0 failed | 2026-10-18 (prima delle chiusure H3/serve: rifinalizzata sotto) |
| `uv run pytest -m real_model` (pesi reali) | ✅ 3 passed + 1 failed ATTESO (gate audio come prova) + H3 transcription real passed | su venv lockato; DB di test in tmp/am-replay-* dedicati |
| `uv run python scripts/evaluate.py --split test --backend real --report docs/evaluation/final.json` | ✅ S1/S2t/S3t/S4t/S5t PASS | commit+lockfile+hardware+ablation nel report |
| Smoke e2e reale headless (`scripts/smoke_headless.sh`) | ✅ runtime_up→snapshot→suggerimento live→teardown 0 orfani | log in `/tmp/am-smoke-launcher.out`; riproducibile |
| Revisore fresco (plan-walkthrough) | ✅ 9 findings → bloccanti risolti (schema report, gate qualificati _text), consigliabili risolti o tracciati | questo registro + 08/07 note datate |
| Checklist manuale GNOME (`docs/evaluation/manual.md`) | ⏳ **manuale non eseguito** | richiede desktop utente + `sudo dnf install -y mpv` (sudo non disponibile all'orchstrazione) |

## Presenza esecutori e chiusure

| Esecutore | Esito | Verifica Orchestrator |
|---|---|---|
| Agent-1-Storage | ✅ TASK_DONE ST1–ST4 + delta CLI bank | 51 test + ruff + CLI rieseguiti; import bank reale verificato |
| Agent-2-Inference | ✅ Fase 1 chiusa + serve + fix | 58 test; probe run-1/2; serve verificato nello smoke reale |
| Agent-3-Ingestion | ✅ CHIUSO su verifica diretta + tempi riportati | 57+ test, fix TIMEOUT, CLI, run reale verificato: health 24.3 s, scan#1 11.8 s (16 idx), scan#2 6.1 s (baseline) → ~0.22 s/chunk reale, zero timeout; TASK_DONE×2 consegnati ma non recapitati (nota canale registrata) |
| Agent-4-Sensors | ✅ TASK_DONE L1–L4 + latency_sec | 95 test; harness usato da gate finale e S5 |
| Agent-5-HUD | ✅ CHIUSO H1–H4 (fix mute incluso e verificato) | 83 test (incl. real_model) + export local senza warning; sola verifica desktop manuale pendente |

## Bug UI HUD (sessione manuale 2026-10-24)

| Bug | Stato |
|---|---|
| Bottoni Pin/Apri delle card inerti (click → flash, nessuna azione) | ✅ **CHIUSO con verifica desktop** (2026-10-24): causa = rebuild lista card ogni 100 ms (`_on_sub_tick`) che distruggeva i bottoni fra press e release; fix = `HudLogic` separata dai widget + rebuild solo su cambio reale; 93 test + ruff verificati; utente ha pinnato e aperto dal vivo con successo (pin nel tray+snapshot, visualizzatore lanciato, nessun flash) |
| Tray visivamente disallineato dopo comandi controllo esterni | ✅ chiuso (stessa causa, stesso fix) |
| Backend pin/open: SANO | verificato via socket controllo: pin ok, arco 0.25 simmetrico nel DB, comandi apertura corretti (evince --page-index=2, mpv --start=40.0) |

## Gate O4

- Fase 1: testo PASS pieno / audio FAIL documentato (decisione utente text-only).
- Calibrazione: griglia normativa senza candidati → estensione cosine a 0.70
  (più severa, criteri .80/.80 invariati, scelta pre-finale, nota datata in 08)
  → congelata `cosine/0.70/0.70/γ.6` (`data/calibration.json`, checksum+spazio vincolati).
- Gate finale text: 8/8 hit rank 1, astensione 5/5, ablation coseno 1.0 / zscore 0.81.
- Smoke reale headless: completo (bootstrap → health → watcher/runner → snapshot →
  suggerimento live → teardown senza orfani).

## Sessione manuale GNOME 2026-10-24 (utente live)

Verificato dal vivo: canale testo (f01.md 0.844, canary f05.pdf), pin doppio con
arco 0.25 simmetrico, aperture xdg-open/evince pag.2/mpv sec.40 coi click, gate
audio run-3 con voce reale (4/4). **Bug trovati e chiusi**: bottoni Pin/Apri
(Agent-5: rebuild 100ms), label Mic fissa (Agent-5: mic_label), **soglie default
al posto della calibrazione in run_poc.py** (Orchestrator: autocarico
data/calibration.json — le card video erano scartate da min_final 1.0 zscore).
## Registro blocker/decisioni (sintesi finale)

1. 2026-10-18 FIX fixture v1.1 (confound voce TTS) dopo run-1 FAIL audio.
2. 2026-10-18 run-2 FAIL audio → FAIL REALE S2-audio → decisione utente text-only.
3. 2026-10-18 griglia calibrazione estesa a 0.70 (nota datata in 08 §3).
4. 2026-10-18 torchvision aggiunta al lock (processor video all'import).
5. 2026-10-18 av pinnata 15.0.x per faster-whisper (extra asr).
6. 2026-10-18 launcher: serve invocazione + sys_python senza resolve() (venv).
7. Gap residuo: `mpv` non installabile senza sudo → voci manuali video dichiarate
   `manuale non eseguito`; ASR su TTS dichiarata non affidabile in qualità (H3 README).
