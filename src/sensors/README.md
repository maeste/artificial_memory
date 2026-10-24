# Sensori e sessione live (Agent-4)

Modulo `src/sensors/` — piano [05_AGENT_4_SENSORS](../../untracked_doc/05_AGENT_4_SENSORS.md),
contratti [07_TECHNICAL_CONTRACTS](../../untracked_doc/07_TECHNICAL_CONTRACTS.md) §4–6, PRD Fase 4.

## Componenti

| File | Responsabilità |
|---|---|
| `ema_tracker.py` | `EMATracker`: una EMA normalizzata per sorgente, validazione vettori, reset |
| `agent_sensor.py` | `parse_agent_line` (plain / claude-jsonl) e `AgentLogTail`: tail incrementale, ANSI/repaint, debounce, rotazione/troncamento |
| `mic_sensor.py` | `VadSegmenter` (RMS, calibrazione 1 s), `MicSensor` (WAV univoci), `AudioCapture` (sounddevice → parec) |
| `session.py` | *vedi nota:* la logica di sessione vive in `runner.py` (`RunnerService`) |
| `runner.py` | `combine_results` (fusione), `RunnerService` (stato/pin/comandi), `LiveInputQueue`, `RunnerServer` (PUB/REP/worker), CLI |
| `replay.py` | `ReplayHarness`: valutazione sui producer e worker reali, CLI |
| `fakes.py` | Fake deterministici dei contratti §4 per i test (Storage/Inference reali arrivati dopo: integrati nel wiring CLI) |
| `test_sensors.py`, `test_session.py` | Test L1–L4, nessun hardware richiesto |

Nota deviazione: il piano prevedeva `session.py` separato; la logica di sessione
(storia, limiti, pin, lease, reset) è dentro `RunnerService` in `runner.py` perché
ne condivide ogni struttura dati. `fakes.py` è aggiuntivo rispetto alla tabella.

## Comandi

Test (dal root del repo):

```bash
uv run pytest src/sensors -q                                   # tutto il modulo
uv run pytest src/sensors/test_sensors.py -k 'ema or fusion or ttl' -q   # L1
uv run pytest src/sensors/test_sensors.py -k agent -q          # L2
uv run pytest src/sensors/test_sensors.py -k mic -q            # L3
uv run pytest src/sensors/test_session.py -q                   # L4 sessione/comandi/IPC
```

Runner live (come lo avvia `src/launcher.py`):

```bash
uv run python -m src.sensors.runner --config data/config.json \
  --agent-log /percorso/assoluto/session.log --agent-format plain
# opzioni: --no-mic, --show-all (salta soglie, DEBUG), --replay-log (fixture: leggi dall'inizio)
```

Replay di valutazione (consumato da `scripts/evaluate.py`):

```bash
uv run python -m src.sensors.replay --config data/probe-config.json \
  --manifest tests/fixtures/ambient/manifest.json \
  --split calibration --source agent_text --repeat 3
```

## Collegare una sessione agente reale — bridge definitivo (verifica congiunta, text-only)

Procedura definitiva per la verifica congiunta con l'Orchestratore (gate PRD
Fase 4, «un messaggio in una sessione agente reale produce suggerimenti»).
Decisione utente: verifica **text-only** — il runner parte con `--no-mic`.

**Terminale 1 — bridge PTY** (il terminale in cui si vuole lavorare; tutto il
traffico del tool finisce nel log):

```bash
mkdir -p /home/maeste/project/artificial_memory/logs
script -q -f -c 'claude' /home/maeste/project/artificial_memory/logs/agent-session.log
```

**Terminale 2 — runner** (il launcher usa questi stessi argomenti):

```bash
uv run python -m src.sensors.runner \
  --config /home/maeste/project/artificial_memory/data/config.json \
  --agent-log /home/maeste/project/artificial_memory/logs/agent-session.log \
  --agent-format plain \
  --no-mic
```

**Verifica**: nel terminale 1 si invia un nuovo messaggio all'agente; il tail
lo rileva (debounce 1 s), il runner lo usa come query e pubblica l'evento
`live.state` con i ricordi. Controllo dal vivo con la CLI condivisa:

```bash
uv run python -m src.control \
  --runtime "$XDG_RUNTIME_DIR/artificial-memory/<run_id>" snapshot
```

Da osservare nel gate: il ricordo pertinente nei risultati, nessuna query
dallo storico preesistente all'avvio (il tail parte dal fondo), pulizia
dell'input nel testo consegnato (ANSI/repaint rimossi). Pulizia e assenza di
query dallo storico sono coperte anche dai test L2. Sostituire `claude` con il
tool realmente installato è un esempio di bridge PTY, non una dichiarazione di
supporto già provato. Per un log strutturato usare il suo path esplicito e
`--agent-format claude-jsonl` (es. `~/.claude/projects/<progetto>/<sessione>.jsonl`).
Il runner non legge mai indiscriminatamente tutte le sessioni utente: solo il
path esplicito ricevuto. L'avvio legge dal fondo del file: la conversazione
storica non diventa query live.

Nota sul canale audio: la pipeline mic (VAD, segmentazione, WAV, replay audio)
resta implementata e dimostrabile a livello unitario e di integrazione mock
(`uv run pytest src/sensors -k 'mic or Vad or audio' -q`); ai gate di
valutazione il canale audio è dichiarato FAIL per decisione utente
(eccezione S2), quindi la calibrazione O4 e il final gate T1–T8 usano solo
`--source agent_text`.

## Microfono

Cattura PCM mono 16 kHz, blocchi 30 ms/480 campioni int16. Backend: prima
`sounddevice` (PortAudio); se fallisce e `parec` è presente, subprocess
`parec --rate=16000 --channels=1 --format=s16le`. Il backend e il device
effettivi compaiono nei log di avvio. Nessun fallback finto: senza hardware
reale il runner prosegue solo con la sorgente testuale.

VAD energetico (RMS): calibrazione 1 s in silenzio, soglia iniziale
`max(0.008, 3×noise_rms)`, pavimento aggiornato solo su frame non vocali.
Segmenti: minimo 600 ms di parlato, chiusura dopo 600 ms di silenzio, massimo
12 s (emissione forzata con overlap 300 ms nel segmento successivo), i troppo
brevi sono eliminati. **Limite dichiarato**: il VAD energetico in ambienti
rumorosi può troncare o trascinare code di parlato; non è un KWS semantico.

Pausa mic (comando `set_mic` o `--no-mic`): chiude la cattura con flush
controllato del segmento pendente; la sorgente testuale resta attiva.

## Comportamenti garantiti (contratti §5/§6)

- Una sorgente non modifica l'EMA dell'altra; scadenza TTL 120 s ricalcolata
  anche su heartbeat; nessuna sorgente attiva → results vuoto, tray invariato.
- Fusione: `top_k×4` candidati per sorgente, unione per memory ID con punteggi
  della sorgente vincente (parità → `agent_text`), `matched_sources` completo,
  soglie applicate dopo l'unione, ordine final/coseno/ID, limite `top_k`.
- Input registrato nella storia prima dell'inferenza: resta esportabile anche
  se l'embedding fallisce; il fallimento aggiorna solo la diagnostica.
- Pin: rinforzo verso i soli pin superstiti, snapshot di MemoryView + ultimi 3
  input; doppio pin idempotente; retry con stesso `command_id` usa la risposta
  memorizzata; DB in errore → pin non aggiunto, risposta `STORAGE_ERROR`.
- Un pin il cui ricordo è cancellato resta nel tray ma non blocca né rinforza.
- `new_session`: nuovo UUID, tracker/storia/pin azzerati, sequence crescente,
  lease 120 s che rinvia la rimozione della directory audio precedente.
- Sessione: limiti 1 h / 2 000 input / 120 MiB audio; al primo limite i nuovi
  input sono sospesi (`sensor_status.session=full`), quelli presenti restano,
  pin/export/reset restano disponibili.
- Eventi PUB: snapshot completi con `source_last_seen` e `dropped_inputs`;
  heartbeat e `index_generation()` ogni 2 s best effort fra richieste.

## Retention e limiti

- WAV live e messaggi di sessione vivono in `<run_dir>/<session_id>/`; reset o
  arresto li eliminano (la rimozione post-reset attende il lease di 120 s).
- Coda input: massimo 8; quando piena scarta il più vecchio, incrementa
  `dropped_inputs` ed elimina il suo WAV non referenziato.
- Nessun ASR nel percorso live; i WAV nativi sono query. Nessuna indicizzazione
  automatica del live nell'archivio.
- La CLI `--replay-log` è diagnostica: legge un log-fixture dall'inizio.
- Il tokenizer del replay è il loader condiviso dell'Ingestion
  (`src.ingestion.watcher.load_tokenizer`: mock a caratteri, reale via
  transformers sulla revisione dello SpaceSpec) — nessuna duplicazione locale.

## Smoke hardware (manuale, marcato `hardware`)

```bash
uv run python -m src.sensors.runner --agent-log "$PWD/logs/agent-session.log" --agent-format plain
# in un altro terminale:
uv run python -m src.control --runtime "$XDG_RUNTIME_DIR/artificial-memory/<run_id>" snapshot
```

Verifiche eseguite in sviluppo: vedi `docs/implementation-status.md` e il
rapporto Lince all'Orchestratore ([TASK_DONE]).


## Limiti nota (2026-10-24, fix tail NUL/banner)

- Il gate `MIN_REAL_CONTENT` (4 alfanumerici sul blocco aggregato) scarta anche
  query genuine molto brevi (<4 caratteri): tradeoff accettato — le query live
  realistiche sono frasi.
- Un prompt-shell reale ("maeste@fedora:~$") ha abbastanza alfanumerici da
  passare il gate: se il rumore-prompt tornasse a inquinare il bridge, la
  direzione è il riconoscimento del pattern-prompt, NON l'aumento della soglia
  (che penalizzerebbe query brevi legittime).
