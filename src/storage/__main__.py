"""CLI autonoma dello storage: import di fixture vettoriali e ricerca da terminale.

Comandi (piano ST4):
    uv run python -m src.storage import --db PATH --manifest PATH
    uv run python -m src.storage search --db PATH --query-vector PATH --top-k 5

Il manifest JSON è accettato in due forme esatte:
- forma CLI: {"space_spec": <asdict SpaceSpec>, "records": [{memory, embedding}, ...]};
- forma bank Fase 1: {"schema_version": 1, "space": <asdict SpaceSpec>, "records": [...]}
  (data/vector-bank-real.json è importabile così com'è).
La query JSON contiene `space_id` e `vector`: lo spazio viene
verificato contro il DB prima di interrogare. L'import valida TUTTO il
manifest prima di toccare il DB, poi raggruppa per file e usa replace_file
(atomicità per file, file in ordine stabile).
"""

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, fields
from pathlib import Path

from src.contracts import SCHEMA_VERSION, MemoryRecord, MemoryView, SearchSettings, SpaceSpec, validate_vector
from src.storage.db import BrainStorage


def _decode_manifest(raw: str) -> tuple[SpaceSpec, dict[str, list[MemoryRecord]]]:
    """Valida l'intero manifest e raggruppa i record per file; nessuna scrittura qui."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MANIFEST_INVALIDO: JSON illeggibile: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MANIFEST_INVALIDO: radice non è un oggetto")
    keys = set(payload)
    if keys == {"space_spec", "records"}:
        spec_raw = payload["space_spec"]
    elif keys == {"schema_version", "space", "records"}:
        # forma bank della Fase 1: stessa identità, envelope diverso
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"SCHEMA_VERSION_MISMATCH: bank schema {payload['schema_version']!r}, attesa {SCHEMA_VERSION}"
            )
        spec_raw = payload["space"]
    else:
        raise ValueError("MANIFEST_INVALIDO: attese le chiavi space_spec e records (o forma bank space/records)")
    if not isinstance(spec_raw, dict) or set(spec_raw) != {f.name for f in fields(SpaceSpec)}:
        raise ValueError("MANIFEST_INVALIDO: space_spec non conforme al contratto")
    spec = SpaceSpec(**spec_raw)
    view_keys = {f.name for f in fields(MemoryView)}
    by_file: dict[str, list[MemoryRecord]] = {}
    for item in payload["records"]:
        if not isinstance(item, dict) or set(item) != {"memory", "embedding"}:
            raise ValueError("MANIFEST_INVALIDO: record non conforme al contratto")
        memory_raw = item["memory"]
        if not isinstance(memory_raw, dict) or set(memory_raw) != view_keys:
            raise ValueError("MANIFEST_INVALIDO: memory non conforme al contratto")
        record = MemoryRecord(MemoryView(**memory_raw), item["embedding"])
        try:
            validate_vector(record.embedding)
        except ValueError as exc:
            raise ValueError(f"MANIFEST_INVALIDO: vettore del record {memory_raw.get('id')!r}: {exc}") from exc
        by_file.setdefault(record.memory.file_path, []).append(record)
    for file_path, group in by_file.items():
        if len({record.memory.source_sha256 for record in group}) != 1:
            raise ValueError(f"MANIFEST_INVALIDO: source_sha256 incoerenti fra i chunk di {file_path}")
    return spec, by_file


def _read_space_spec(db_path: str) -> SpaceSpec:
    """Legge lo SpaceSpec registrato; serve alla search, che riceve solo space_id."""
    if not os.path.exists(db_path):
        raise ValueError(f"DB_ASSENTE: {db_path} non esiste")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'space'").fetchone()
    if row is None:
        raise ValueError("DB_NON_INIZIALIZZATO: metadata assente")
    return SpaceSpec(**json.loads(row[0]))


def _cmd_import(args: argparse.Namespace) -> int:
    try:
        spec, by_file = _decode_manifest(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    try:
        storage = BrainStorage(args.db, spec)
    except ValueError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    total = sum(len(group) for group in by_file.values())
    try:
        for file_path in sorted(by_file):
            group = by_file[file_path]
            storage.replace_file(file_path, group[0].memory.source_sha256, group)
    except ValueError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    finally:
        storage.close()
    print(
        json.dumps(
            {
                "imported_files": len(by_file),
                "imported_memories": total,
                "space_id": spec.space_id,
                "db": args.db,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    try:
        query = json.loads(Path(args.query_vector).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERRORE: query illeggibile: {exc}", file=sys.stderr)
        return 1
    if not isinstance(query, dict) or set(query) != {"space_id", "vector"}:
        print("ERRORE: la query deve contenere esattamente space_id e vector", file=sys.stderr)
        return 1
    try:
        validate_vector(query["vector"])
    except ValueError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    try:
        spec = _read_space_spec(args.db)
    except (ValueError, json.JSONDecodeError, sqlite3.DatabaseError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    if query["space_id"] != spec.space_id:
        print(
            f"ERRORE: SPACE_MISMATCH: query {query['space_id']} != DB {spec.space_id}",
            file=sys.stderr,
        )
        return 1
    try:
        storage = BrainStorage(args.db, spec)
    except ValueError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    try:
        results = storage.hybrid_search(query["vector"], SearchSettings(top_k=args.top_k))
    except ValueError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
    finally:
        storage.close()
    print(json.dumps({"results": [asdict(item) for item in results]}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.storage", description="Archivio persistente: import e ricerca locale"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cmd_import = sub.add_parser("import", help="importa un manifest di MemoryRecord (JSON)")
    cmd_import.add_argument("--db", required=True, help="path del DB SQLite")
    cmd_import.add_argument("--manifest", required=True, help="manifest JSON con space_spec e records")
    cmd_import.set_defaults(func=_cmd_import)

    cmd_search = sub.add_parser("search", help="ricerca ibrida con un vettore query (JSON)")
    cmd_search.add_argument("--db", required=True, help="path del DB SQLite")
    cmd_search.add_argument("--query-vector", required=True, help="JSON con space_id e vector")
    cmd_search.add_argument("--top-k", type=int, default=5, help="massimo numero di risultati (default 5)")
    cmd_search.set_defaults(func=_cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
