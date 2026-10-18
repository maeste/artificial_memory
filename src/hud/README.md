# HUD — apertura, contesto e trascrizione (Agent-5)

Moduli di proprietà Agent-5-HUD: card/tray GTK4 (H4), aperture localizzate,
export del contesto e trascrizione ASR locale opzionale. Gate headless:

```sh
uv run pytest src/hud -m 'not desktop and not real_model' -q
```

## Apertura dei ricordi

`openers.build_open_command(memory)` costruisce argv senza shell: MP4
`mpv --start=<secondi> file` (decimali conservati), PDF
`evince --page-index=<pagina> file` (numero pagina utente, 1-based), altri
formati `xdg-open file`. `open_memory_file` verifica l'esistenza del file e
la disponibilità dello strumento prima di `subprocess.Popen(argv)`.

Preflight strumenti: `openers.missing_open_tools()` → `['mpv']` se manca il
player video (su questa macchina è il caso: il gate manuale GNOME richiede
l'installazione di mpv; i test costruiscono argv senza eseguirlo).

## Export del contesto

`context_exporter.assemble_and_export_context(state, session_inputs,
export_root, settings)` produce `exports/<session_id>/<export_id>/` con
`context.md`, `manifest.json`, `session-context.json`, `assets/`,
`memories/` e (solo in local) `transcripts/`. Lo staging nasce nella stessa
directory e diventa il pacchetto con rename atomico; un errore fatale
rimuove lo staging e solleva `ExportFailedError('EXPORT_FAILED: ...')`.

- Budget Markdown: `settings.context_max_chars` (default 24.000, minimo
  12.000). Troncamento solo su confini di paragrafo, con omissioni elencate
  nel manifest. Se il contenuto obbligatorio non entra: `EXPORT_FAILED`.
- Sorgenti: ogni allegato nasce da una copia con SHA-256 verificato contro
  `memory.source_sha256`; sorgente cambiata o assente ⇒ «originale
  selezionato indisponibile», testo congelato conservato, niente caption.
- Video: clip solo dell'intervallo selezionato, con seek accurato e
  re-encode ffmpeg; la clip parte da zero, gli estremi originali restano
  nel manifest. Immagini e pagine scansionate: un agente solo testuale non
  ne legge il contenuto (dichiarato nel Markdown).
- Clipboard: `wl-copy` se presente; assenza o errore non invalida il
  pacchetto (`clipboard_status` in manifest ed `ExportResult`).

## Trascrizione ASR: `off` (default) / `local`

La scelta vive in `ExportSettings.transcription` (contratti §3), si congela
all'avvio di ogni export e vale solo per quello successivo.

**off** — zero import, download o chiamate ASR: funziona senza l'extra
`asr` installato. I WAV e le clip restano allegati; il Markdown dichiara
che occorre un agente capace di ascoltare gli allegati.

**local** — faster-whisper CPU int8, solo sui WAV selezionati e sull'audio
dei video pinnati (mai tutto l'archivio). Le clip senza traccia audio non
vengono passate all'ASR (restano clip visuali: stato `no_audio` nel
manifest, nessun falso errore). Parametri operativi deterministici
registrati nel manifest: `language='it'` (configurabile), `beam_size=1`,
`temperature=0`, `condition_on_previous_text=False`. Le trascrizioni sono
testo del modello ASR, non citazioni garantite. Budget
`settings.max_transcribe_sec` (default 300 s): prima l'audio dei pin, poi
il parlato recente; gli ID oltre budget sono dichiarati in
`manifest.transcription.excluded_ids` e l'audio resta allegato — mai
troncamento silenzioso di una frase. Cache per hash WAV + revisione ASR +
lingua + parametri.

Preparazione esplicita del modello (una volta, mai dal click export):

```sh
uv sync --extra asr
uv run python -m src.hud.transcriber prepare --model base --output ./data/models/asr
```

Il comando risolve il commit corrente di `Systran/faster-whisper-base`,
scarica quella revisione e scrive `manifest.json` con modello/revisione/
versioni. `transcriber.preflight_local(model_dir)` verifica directory,
manifest e pesi **senza rete**: modello mancante richiede una nuova
preparazione, non un download automatico. Avvio con trascrizione locale:
`--transcription local --asr-model-dir ./data/models/asr`.

Errori ASR (modello assente, trascrizione fallita, output vuoto): warning
esplicito, allegati conservati, stato per input `unavailable`/`failed`/
`empty` nel manifest. Nessun fallback a servizi remoti.

Nota di verifica (decisione utente, 2026-10-18): il canale live **testuale**
è quello verificato semanticamente (gate S2); il microfono resta
implementato fuori gate — la trascrizione ASR serve l'export, non prova la
qualità del retrieval vocale.

Limiti misurati del modello ASR (smoke reale, whisper-base @ ebe41f70 su
voce sintetica espeak it+f4 dei fixture): C2 e C3 parzialmente riconosciute
("Mostrami", "grafico", "simbolo", "esclamativo"); C1 sostanzialmente
 illeggibile; le negative vanno in loop allucinatorio ("niente" ripetuto).
Il test `real_model` è quindi uno smoke di pipeline (termini sopravvissuti
del testo fixture su C2/C3), non una prova di qualità della trascrizione.

## Avvio desktop (H4)

```sh
# HUD collegato al runtime attivo (endpoints letti da state.json)
uv run python -m src.hud.app --runtime "$XDG_RUNTIME_DIR/artificial-memory/<run_id>"

# Demo senza backend: publisher fake, nessun socket di controllo
uv run python -m src.hud.app --demo

# Con trascrizione locale preparata (v. sezione precedente)
uv run python -m src.hud.app --runtime <run_dir> --transcription local --asr-model-dir ./data/models/asr
```

Finestra Adw ~380×720 ridimensionabile, tema di sistema, mai sempre in
primo piano. Header: backend, stato mic/agente, scarti; card con titolo,
estratto (3 righe), formato, sorgenti, score e badge boost; tray dei pin
con rimozione; Mic on/off, Nuova sessione (la storia temporanea riparte,
pin ed export restano), selector trascrizione e bottone «Prepara contesto».
L'export gira su worker: la finestra resta reattiva, il bottone è
disabilitato fino al termine, toast con percorso, clipboard e warning.
Il cambio selector vale per l'export successivo. Scorciatoia locale
Ctrl+Shift+C facoltativa; nessuna hotkey globale (D14).

Verifica GNOME: checklist manuale in `docs/evaluation/manual.md`
(copia operativa di `untracked_doc/08_EVALUATION.md` §6: avvio tardivo,
pin, unpin, grafo, apertura, entrambe le modalità export) — richiede
display attivo e resta da compilare; non coperta dal gate headless.

## Test

- `test_hud.py` — aperture e logica card senza display.
- `test_export.py` — pacchetto, budget, lease, allegati media; sezione
  trascrizione con spy (il test `real_model` confronta i termini attesi su
  `tests/fixtures/ambient/query-C1.wav` e `query-C2.wav` e richiede l'extra
  `asr` più il modello preparato).

Verifica GNOME (H4): checklist manuale in `docs/evaluation/manual.md`,
da eseguire con display attivo.
