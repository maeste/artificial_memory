# src/ingestion — archivio alimentato dalla directory

Trasforma i file supportati nella directory osservata in ricordi localizzabili
(PRD Fase 3) e mantiene l'archivio aggiornato: scansione iniziale, watcher,
modifica/cancellazione/spostamento. Scansione e watcher condividono la stessa
`index_file`.

## Formati supportati

| Formato | Estrazione | Chunk |
|---|---|---|
| `.md`, `.markdown` | split per heading ATX fuori da code fence; testo prima del primo heading = sezione `Introduzione` | `section:<gerarchia>:<occorrenza>:part:<indice>` |
| `.pdf` | testo per pagina (pagine da 1); pagina con immagini o meno di 80 caratteri utili → render PNG 120 DPI, lato max 1600 px, in staging, insieme al testo | `page:<numero>:part:<indice>` |
| `.png`, `.jpg`, `.jpeg` | decode Pillow validato; copia RGB in staging solo se la modalità non è RGB | `image:0` |
| `.mp4` | ffprobe (durata traccia video), finestre 45 s stride 40 s, 3 frame al 10/50/90 % + WAV mono PCM 16 kHz per finestra (senza traccia audio: `audio_path=None`) | `video:<start_ms>:<end_ms>` |

Limiti: sezioni/pagine oltre 1500 token frazionate su confini di paragrafo
(overlap 100 token fra parti della stessa sezione, mai attraverso sezioni);
un paragrafo-muro viene diviso per token con il tokenizer della revisione.
Il corpo completo va in `content_text` e `text_payload` (il titolo non
sostituisce il contenuto); snippet = primi 350 caratteri. Per pagina visuale
senza testo: snippet fattuale «Pagina N, contenuto visuale; allegato richiesto».
Nessuna caption inventata. Nessun troncamento silenzioso.

Identità (contratto §3): `source_sha256` = hash dei byte del file intero;
`content_sha256` = hash di header logico (senza nome file) + newline + corpo
per il testo, di `source_sha256 + chr(0) + chunk_key` per i media; ID da
`make_memory_id` su path canonico. Una rinomina crea una nuova fonte.

## API

```python
from src.ingestion import extract_file_chunks, index_file, scan_directory

pairs = extract_file_chunks(file_path, tokenizer, temp_dir)   # solo estrazione
changed = index_file(file_path, storage, client, tokenizer, temp_root)  # False se hash invariato
results = scan_directory(watch_dir, storage, client, tokenizer, temp_root)
# results: {path: 'indexed' | 'unchanged' | 'deleted' | 'error:CODE'}
```

`index_file` è atomica: doppio hash (iniziale e finale) per potere committare,
`replace_file` solo dopo embedding riuscito di tutti i chunk, staging univoco
rimosso nel `finally`. Un errore lascia la versione precedente intatta. Su
TIMEOUT un solo retry con stesso payload e `request_id`. Un file cambiato
durante il lavoro solleva `SourceChangedError` (il watcher lo riaccoda).

`WatcherService` serializza l'indicizzazione: deduplica coda per path,
debounce 1 s + stabilità size/mtime fra due rilevazioni distanti 1 s (il doppio
hash resta la protezione reale contro le copie concorrenti), riaccodamento se
un evento arriva durante il lavoro, move = rimozione vecchio path + indicizzazione
del nuovo (gli archi della vecchia identità non migrano).

## CLI

```sh
uv run python -m src.ingestion.watcher --config PATH          # scan + watcher
uv run python -m src.ingestion.watcher --config PATH --once   # riconciliazione singola
```

`--config` è il `config.json` scritto da `run_poc.py` nel `run_dir` (space,
paths, timeout). Il watcher usa `paths.watch_dir`, `paths.db_path`,
`paths.embed_endpoint` e `embed_timeout_ms`. Staging in `<run_dir>/staging`,
rimossa a fine di ogni file. La directory osservata è una copia pulita dei
file utente (es. `data/watch-corpus/`): mai puntare `tests/fixtures/ambient/`,
che contiene gli artefatti interni del builder (`_staging/`, query, manifest).

## Stati ed errori

| Stato / codice | Significato |
|---|---|
| `indexed` | nuova versione indicizzata (o versione vuota, per file senza chunk) |
| `unchanged` | hash invariato: zero richieste di inferenza |
| `deleted` | fonte rimossa dal DB (sparita dalla radice) |
| `error:INVALID_MEDIA` | file corrotto o non leggibile (PDF/immagine/video) |
| `error:SOURCE_CHANGED` | sorgente modificata durante l'estrazione: riaccodata |
| `error:UNSUPPORTED_FORMAT` | estensione fuori dalla lista (la scan la filtra già) |
| `error:OUTSIDE_WATCH_ROOT` | fonte nel DB fuori dalla radice configurata: non toccata |
| `error:INGESTION_ERROR` | altro errore (inferenza, vettore non valido, …) |

Un file in errore non blocca gli altri.

## Tokenizer

Il backend `real` carica via transformers il tokenizer della revisione
congelata nello `SpaceSpec` (senza pesi del modello). Il backend `mock` usa la
tokenizer a carattere del modulo (`CharTokenizer`): approssimazione
conservativa dei limiti, deterministica. Un test `real_model` dell'Orchestratore
verificherà i limiti con il tokenizer reale.

## Integrazione con i moduli reali

Le API reali `src.storage.db.BrainStorage` (Agent-1) e
`src.inference.client.EmbedClient` (Agent-2) sono importate dalla CLI al primo
uso. Il riconoscitore di timeout copre la forma reale del client
(`EmbeddingError` con `.code == 'TIMEOUT'`) e quella dei fake di test
(`.error_code`). Il backend `real` carica via transformers il tokenizer della
revisione congelata nello `SpaceSpec`; il `mock` usa la tokenizer a carattere.
Gli errori del client non riconosciuti come TIMEOUT sono definitivi: la
versione precedente resta indicizzata.
