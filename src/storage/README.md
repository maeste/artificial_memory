# Storage — indice persistente e ricerca associativa

Implementazione ST1–ST4 di `untracked_doc/02_AGENT_1_STORAGE.md`; API pubblica
esattamente quella dei contratti §4 (`untracked_doc/07_TECHNICAL_CONTRACTS.md`).
Proprietà esclusiva dell'esecutore storage: nessun altro modulo scrive qui.

## Uso

```bash
# suite completa (41 test: schema/CRUD, atomicità, ranking, CLI)
uv run pytest src/storage -q

# import di un manifest di record vettoriali — due forme accettate:
#   CLI:    {"space_spec": <asdict SpaceSpec>, "records": [{memory, embedding}, ...]}
#   Bank:   {"schema_version": 1, "space": <asdict SpaceSpec>, "records": [...]}
# data/vector-bank-real.json (Fase 1) è importabile così com'è
uv run python -m src.storage import --db data/brain-mock.db --manifest manifest.json

# ricerca con un vettore query JSON {"space_id": "...", "vector": [256 float]}
uv run python -m src.storage search --db data/brain-mock.db --query-vector query.json --top-k 5
```

Risultato della ricerca: `{"results": [...]}` con ogni `ScoredMemory` serializzato
e `memory` (la `MemoryView` completa) annidata. `winning_source` è sempre `null`:
storage non assegna sorgenti live (contratti §5), le soglie di visualizzazione
(`min_final`, filtro UI) non sono applicate qui.

Uso come libreria:

```python
from src.contracts import SearchSettings, SpaceSpec
from src.storage import BrainStorage

db = BrainStorage("data/brain-mock.db", SpaceSpec("mock", "semantic-hash-mock", "1"))
results = db.hybrid_search(query_vector, SearchSettings(top_k=5))
db.close()
```

## Backend vettoriale

`BrainStorage(db_path, space, vector_backend='auto')` sceglie all'apertura:

- `sqlite_vec`: estensione sqlite-vec (0.1.9 nel venv) — i coseni sono calcolati
  dall'estensione su righe complete con `vec_distance_cosine`;
- `numpy`: prodotto matrice-vettore in float64 sui BLOB;
- `auto`: prova sqlite-vec, altrimenti passa a NumPy riportando il fallback negli
  attributi `backend_fallback`/`backend_note` (diagnostica, non API contrattuale);
  `sqlite_vec` esplicito fallisce con `SQLITE_VEC_UNAVAILABLE`.

Il backend resta fisso per tutta la vita dell'istanza. Quando è attivo sqlite-vec,
la tabella virtuale `vec_memories` è un mirror tenuto nella stessa transazione dei
metadati (creazione, aggiornamento, eliminazione esplicita dei vettori rimossi,
rielaborazione all'apertura se il DB nasce con NumPy). Il BLOB in `memories` resta
la fonte di verità per export, test e fallback.

## Schema e transazioni

Tabelle: `metadata` (schema_version, space, index_generation), `files`
(path → sha256, indexed_at), `memories` (id, file_path FK con cascata, modality,
payload_json, embedding BLOB), `synaptic_edges` (coppia → peso, updated_at, entrambe
le direzioni). Pragme: WAL, `foreign_keys=ON`, `busy_timeout=5000`; ogni thread apre
la propria connessione (`isolation_level=None` con `BEGIN IMMEDIATE` esplicito,
rollback su ogni eccezione). Una transazione scrive metadati, testo e vettori
insieme; nessuna inferenza dentro le transazioni.

Vettori: BLOB little-endian float32 di 256 elementi, finiti e normalizzati
(`validate_vector` prima di ogni commit); `payload_json` contiene la `MemoryView`
completa senza embedding duplicati.

Garanzie principali, tutte coperte da test:

- un DB ospita un solo `space_id`: riapertura con spazio diverso →
  `ValueError('SPACE_MISMATCH')` senza cancellare nulla; schema incompatibile →
  `SCHEMA_VERSION_MISMATCH`/`SCHEMA_INCOMPATIBLE`, nuovo DB richiesto;
- `replace_file` è atomico: valida tutti i record, poi in un'unica transazione
  aggiorna il hash del file, inserisce/aggiorna i sopravvissuti, elimina solo gli
  ID obsoleti e incrementa `index_generation`. Un errore lascia vecchi record,
  hash e archi intatti. Gli ID sopravvissuti conservano `created_at` e i propri
  archi: un chunk modificato riceve un nuovo ID e perde soltanto i collegamenti
  propri; una rinomina di file equivale a rimozione + nuova fonte nel PoC;
- sostituzione con lista vuota: non è un errore, conserva la registrazione del
  file col nuovo hash e rimuove i vecchi chunk;
- `reinforce_synapses` rinforza simmetricamente in transazione (delta 0.25 di
  default, tetto 1.0), deduplica, esclude sé stesso, ignora ID rimossi e non
  tocca embedding né generazione;
- `delete_by_filepath` rimuove file e record; gli archi cadono a cascata solo per
  gli ID rimossi; file assente → 0 senza incrementi.

## Ranking (ST3)

Esatto su tutto l'archivio, complessità O(N·d): coseno su vettori unitari, calibrazione
z per gruppo di `modality` (`normalized_scores`: deviazione con floor 0.05 sopra i 3
record, calibrazione fissa `(cos-0.30)/0.15` sotto), `base = z` (zscore) o `cos`
(coseno), semi attivi con `base > activation_threshold` e `cos >= min_cosine`,
`boost = gamma · max(base_seme · peso)` a un solo hop senza somma di versi né
ricircolo, ordinamento final desc, coseno desc, ID crescente, troncato a `top_k`.

`rank(records, query_vec, edges, settings)` resta interno al modulo; i test
verificano separatamente la matematica pura e i pesi letti dal DB.

## Limiti del PoC

- Ricerca esatta, nessun ANN: le prestazioni su archivi grandi non sono promesse né
  misurate; il corpus piccolo è scandito per intero a ogni ricerca.
- Equivalenza fra backend: i coseni sqlite-vec sono calcolati in float32
  (differenza misurata ≤ ~3e-6 dal riferimento NumPy float64). La calibrazione z
  amplifica per 1/std del gruppo: con gruppi ben dispersi i punteggi coincidono
  entro 1e-5 (test dedicato); gruppi quasi piatti (std vicino al floor 0.05)
  possono superare quella tolleranza. Il path NumPy è il riferimento.
- `delete` di un file inesistente e `reinforce` senza candidati validi sono no-op
  committati; l'import CLI è atomico per file, non per manifest.
- Nessuna ripresa automatica dopo crash; il grafo e gli export persistono.
- Mock e dati reali non condividono DB (`brain-mock.db` vs `brain-real.db`).

## Test

```bash
uv run pytest src/storage -q          # 41 passed
uv run ruff check src/storage/        # pulito
```

Test su `tmp_path`, mai sui DB in `data/`. Il backend sqlite-vec è installato nel
venv: i test parametrizzati non accettano skip.
