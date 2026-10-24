"""Package ingestion — proprietà esclusiva dell'esecutore ingestion.

API pubblica (firme del contratto 1.0, 07_TECHNICAL_CONTRACTS §4), nel modulo
`src.ingestion.watcher`:
- index_file(file_path, storage, client, tokenizer, temp_root) -> bool
- scan_directory(watch_dir, storage, client, tokenizer, temp_root) -> dict[str, str]
- WatcherService, wait_stable, run_once, load_tokenizer, CharTokenizer
e nel modulo `src.ingestion.extractors`:
- extract_file_chunks(file_path, tokenizer, temp_dir) -> list[(MemoryView, EmbedRequest)]
- InvalidMediaError, UnsupportedFormatError, IngestionError, SourceChangedError

CLI: uv run python -m src.ingestion.watcher --config PATH [--once]

Nota: qui non si importano i sottomoduli per via del lancio `python -m
src.ingestion.watcher` (il package viene importato prima del modulo come
__main__: re-importarli dall'__init__ produce un doppio caricamento).
"""
