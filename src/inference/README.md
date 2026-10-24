# Inference — client/server embedding multimodale

Proprietà esclusiva di Agent-2-Inference. Server REP proprietario del modello,
client REQ per thread, DTO e wire format di `src/contracts.py` (contratto 1.0),
identità dello spazio verificata su ogni risposta.

## Installazione provata (2026-10-18, Fedora, Python 3.12.9)

| Componente | Versione | Nota |
|---|---|---|
| torch | 2.14.1+cu130 | CUDA attiva, bf16 supportato |
| torchvision | 0.29.1+cu130 | **obbligatoria**: `EmbeddingGemma2VideoProcessor` la importa al load anche senza input video; installata `--no-deps` dall'indice pytorch cu130 |
| transformers | 5.20.1 | |
| sentence-transformers | 6.1.0 | |
| numpy / pyzmq / soundfile | 2.5.4 / 27.2.0 / (venv) | |

Modello: `google/embeddinggemma-2`, revisione HF pinnata
`914f7f89142e33e77833254d9c9b90c3cef7303b` (mai `main` nel DB). Pesi ~1,5 GB in
`~/.cache/huggingface`:

```sh
uv run python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('google/embeddinggemma-2', revision='914f7f89142e33e77833254d9c9b90c3cef7303b'))"
```

SpaceSpec reale: `backend=real, model_id=google/embeddinggemma-2,
revision=914f7f…7303b, dimension=256, preprocessing=eg2-poc-v1`.
Il mock usa `semantic-hash-mock/1/mock` e uno spazio mai confrontabile.

## Componenti

- `client.py` — `EmbedClient(endpoint, expected_space_id, timeout_ms=30000)`:
  `embed(req) -> list[float]`, `health() -> dict`, `close()` idempotente.
  `EmbeddingError(code, message)` con `.code`. Timeout → `TIMEOUT` + REQ
  chiuso con LINGER 0; la richiesta successiva ricrea il socket. Il client non
  ripete mai da solo: il chiamante può riprovare una volta.
- `server.py` — `EmbedServer(endpoint, space, backend_factory)`: load+warmup in
  thread separato, health ready solo dopo warmup riuscito, ogni messaggio ha
  risposta (un errore non rompe recv/send), nessun fallback real→mock.
- `model.py` — `build_model_input` (prefissi e placeholder `<|image|>`/`<|audio|>`
  in ordine), `project_output` (256d unitarie, rifiuta non finiti e norma
  < 1e-12), `budget_check` (6000 token testo, 3 immagini, 45 s audio 16 kHz
  mono, budget totale 8192), `EmbeddingGemmaBackend` (bf16 su CUDA, mai fp16),
  `SemanticHashMock` (solo `--backend mock`, hashing deterministico).
- `__main__.py` — CLI `embed` e `probe`.

## Uso per gli altri esecutori (Sensori, Ingestion, evaluation)

```python
from src.contracts import EmbedRequest
from src.inference.client import EmbedClient

client = EmbedClient(endpoint, expected_space_id=space.space_id, timeout_ms=30000)
vector = client.embed(
    EmbedRequest(request_id="q1", is_query=True, text_payload="cache del database")
)
client.close()
```

Health per il launcher: `client.health()` → `{ready, space_id, backend, error}`;
confrontare `space_id` con quello atteso prima di aprire il DB.

## CLI

```sh
uv run python -m src.inference embed --config data/probe-config.json \
  --text 'query di prova' --query
uv run python -m src.inference embed --config data/probe-config.json \
  --audio tests/fixtures/ambient/query-C1.wav
uv run python -m src.inference probe --config data/probe-config.json \
  --manifest tests/fixtures/ambient/manifest.json --report report.json \
  [--bank bank.json] [--split calibration] [--repeats 3]
```

`embed` stampa JSON con SpaceSpec, richiesta, vettore e latenza, marcati con il
backend. `probe` embedda corpus e query dal manifest senza Storage, ranking
coseno NumPy (vettori unitari), bank `MemoryRecord` per l'import Storage e
report con load/warmup/p50/p95 nearest-rank e hardware. Exit code 2 se il gate
preliminare è applicabile e fallisce. Config attesa:
`{"schema_version": 1, "space": {...}, "device": "auto|cpu|cuda"}`.

### serve (contratto del launcher)

```sh
uv run python -m src.inference serve --config <run_dir>/config.json
```

Servizio REP di lunga durata: legge il config di bootstrap del launcher —
passato INTEGRALMENTE ai figli (07 §6): le chiavi degli altri figli (search,
alpha, session_*, agent_log, ...) sono accettate e ignorate; la strettezza
riguarda i campi consumati (`space`, `paths.embed_endpoint`, verifica di
`space_id` se presente). Binda `EmbedServer` su
`paths.embed_endpoint` e resta vivo fino a SIGTERM/SIGINT. Load+warmup in
thread dedicato; log su stderr (`REP in ascolto`, `modello pronto`,
`arresto completato`). Se il caricamento fallisce: log dell'errore e
**exit 1, mai fallback al mock**. Il launcher attende health (ready) sullo
stesso endpoint fino a `startup_timeout_sec`.

## Test

```sh
uv run pytest src/inference -m 'not real_model' -q   # 53 test, senza pesi/GPU
uv run pytest src/inference -m real_model -q         # richiede i pesi in cache
```

## Stato misurato (RTX 5000 Ada, bf16, revisione pinnata)

- Load 6,8 s; warmup 0,35 s; corpus 26 chunk: p50 26,5 ms, p95 204 ms.
- Query testo p50 25,4 ms / p95 36,1 ms; query audio p50 50,6 ms / p95 72,5 ms.
- Gate preliminare Fase 1 (C1–C4, split calibration): **testo 4/4** (tutti
  rank 1, tutti i formati), **audio 2/4** → gate FALLITO, `[BLOCKER]` inviato.

## Errori noti e stato del gate

- Gate preliminare Fase 1, run-1 (fixture v1.0): testo 4/4 (tutti rank 1),
  audio 2/4 (C1 gold rank 11/26, C2 rank 12/26). Ipotesi del confound voce
  TTS (query e narrazioni con lo stesso espeak-ng) confermata come anomalia
  delle fixture: correzione v1.1 dell'Orchestratore (voce query "it+f4",
  corpus byte-identico). Rapporto run-1: `docs/evaluation/probe-failed-run1.json`.
- Gate preliminare Fase 1, run-2 (fixture v1.1, senza confound): testo 4/4
  invariato (MRR 1.0), **audio 2/4**: C3 rank 4, C4 rank 3, C1/C2 fuori top5;
  hit@5 audio 0.5, MRR 0.146. Il fallimento S2-audio è REALE, registrato
  come evidenza secondo il criterio del gate: nessuna ulteriore modifica di
  fixture/query/gold/soglie. Rapporto run-2: `docs/evaluation/probe-failed-run2.json`.
  Il test `real_model` del gate resta rosso come prova del fallimento.
- Dipendenza `torchvision` non dichiarata dal modello nel README upstream ma
  richiesta al load: segnalata per il lockfile.
- `SemanticHashMock` non supera la Fase 1 e non attiva mai il fallback: serve
  solo a sviluppare cablaggio con DB e SpaceSpec dedicati.
## Canale audio: voce naturale vs sintetica (2026-10-24)

Il gate Fase 1 è PASS (run-3) con **voce reale**: parlato italiano naturale
allinea bene al testo dei documenti (cos gold 0.70–0.83, 4/4, tutti i formati).
Il parlato sintetico espeak-ng è **fuori dominio per l'encoder**: le query TTS
mancavano i gold testuali (run-1/run-2, FAIL conservate in
`docs/evaluation/probe-failed-run{1,2}.json`). Implicazione operativa: le
fixture audio sintetiche non vanno usate per giudicare la qualità audio→testo
di questo modello; il percorso microfono (voce umana) è il caso d'uso reale.
