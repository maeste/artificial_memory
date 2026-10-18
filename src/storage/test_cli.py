"""Test ST4 — CLI autonoma: import, ricerca, riapertura, mismatch e DB vuoto."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.contracts import MemoryRecord, MemoryView, SpaceSpec, make_memory_id
from src.storage.__main__ import main
from src.storage.db import BrainStorage

MOCK_SPEC = SpaceSpec("mock", "semantic-hash-mock", "1")
QUERY = [1.0] + [0.0] * 255


def aligned_vector(cos):
    rest = math.sqrt(max(0.0, 1.0 - cos * cos) / 255.0)
    return [cos] + [rest] * 255


def make_record(path, key, cos, modality="markdown"):
    view = MemoryView(
        id=make_memory_id(path, key, "a" * 64),
        file_path=path,
        modality=modality,
        chunk_key=key,
        content_sha256="a" * 64,
        source_sha256="b" * 64,
        context_header=f"{path} > {key}",
        snippet_text=key,
        content_text=key,
        created_at=0.0,
    )
    return MemoryRecord(view, aligned_vector(cos))


def write_manifest(path: Path, spec: SpaceSpec, records: list[MemoryRecord]) -> Path:
    path.write_text(
        json.dumps(
            {
                "space_spec": asdict(spec),
                "records": [{"memory": asdict(record.memory), "embedding": record.embedding} for record in records],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def write_query(path: Path, space_id: str) -> Path:
    path.write_text(json.dumps({"space_id": space_id, "vector": QUERY}), encoding="utf-8")
    return path


def test_import_then_search_roundtrip(tmp_path, capsys):
    db = str(tmp_path / "brain.db")
    records = [
        make_record("/tmp/a.md", "k:a1", 0.9),
        make_record("/tmp/a.md", "k:a2", 0.4),
        make_record("/tmp/b.md", "k:b1", 0.7),
    ]
    manifest = write_manifest(tmp_path / "manifest.json", MOCK_SPEC, records)

    assert main(["import", "--db", db, "--manifest", str(manifest)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["imported_files"] == 2
    assert summary["imported_memories"] == 3
    assert summary["space_id"] == MOCK_SPEC.space_id

    query = write_query(tmp_path / "query.json", MOCK_SPEC.space_id)
    assert main(["search", "--db", db, "--query-vector", str(query), "--top-k", "2"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    assert len(results) == 2
    assert set(results[0]) == {
        "memory",
        "cosine_score",
        "z_score",
        "synaptic_boost",
        "final_score",
        "matched_sources",
        "winning_source",
    }
    assert results[0]["memory"]["id"] == records[0].memory.id
    assert results[0]["cosine_score"] == pytest.approx(0.9, abs=1e-5)
    assert results[0]["winning_source"] is None

    reopened = BrainStorage(db, MOCK_SPEC, "numpy")
    try:
        assert len(reopened.all_memories()) == 3
        assert reopened.index_generation() == 2
    finally:
        reopened.close()


def test_search_on_empty_db(tmp_path, capsys):
    db = str(tmp_path / "brain.db")
    seed = BrainStorage(db, MOCK_SPEC, "numpy")
    seed.close()

    query = write_query(tmp_path / "query.json", MOCK_SPEC.space_id)
    assert main(["search", "--db", db, "--query-vector", str(query)]) == 0
    assert json.loads(capsys.readouterr().out) == {"results": []}


def test_search_rejects_space_mismatch(tmp_path, capsys):
    db = str(tmp_path / "brain.db")
    records = [make_record("/tmp/a.md", "k:a1", 0.9)]
    manifest = write_manifest(tmp_path / "manifest.json", MOCK_SPEC, records)
    assert main(["import", "--db", db, "--manifest", str(manifest)]) == 0
    capsys.readouterr()

    query = write_query(tmp_path / "query.json", "f" * 64)
    assert main(["search", "--db", db, "--query-vector", str(query)]) == 1
    assert "SPACE_MISMATCH" in capsys.readouterr().err


def test_search_missing_db(tmp_path, capsys):
    query = write_query(tmp_path / "query.json", MOCK_SPEC.space_id)
    assert main(["search", "--db", str(tmp_path / "assente.db"), "--query-vector", str(query)]) == 1
    assert not (tmp_path / "assente.db").exists()  # nessun file creato per errore


def test_import_rejects_space_mismatch_on_existing_db(tmp_path, capsys):
    db = str(tmp_path / "brain.db")
    seed = BrainStorage(db, MOCK_SPEC, "numpy")
    seed.close()

    other_spec = SpaceSpec("mock", "altro-modello", "1")
    manifest = write_manifest(tmp_path / "manifest.json", other_spec, [make_record("/tmp/a.md", "k:a1", 0.9)])
    assert main(["import", "--db", db, "--manifest", str(manifest)]) == 1
    assert "SPACE_MISMATCH" in capsys.readouterr().err

    reopened = BrainStorage(db, MOCK_SPEC, "numpy")
    try:
        assert reopened.all_memories() == []
        assert reopened.index_generation() == 0
    finally:
        reopened.close()


def test_import_rejects_invalid_vector_before_touching_db(tmp_path, capsys):
    db = str(tmp_path / "brain.db")
    record = make_record("/tmp/a.md", "k:a1", 0.9)
    bad = MemoryRecord(record.memory, [0.5] * 256)  # norma invalida
    manifest = write_manifest(tmp_path / "manifest.json", MOCK_SPEC, [bad])

    assert main(["import", "--db", db, "--manifest", str(manifest)]) == 1
    assert "MANIFEST_INVALIDO" in capsys.readouterr().err
    assert not Path(db).exists()  # validazione prima della creazione del DB


def test_import_rejects_incoherent_manifest_shapes(tmp_path, capsys):
    db = str(tmp_path / "brain.db")

    (tmp_path / "senza_record.json").write_text(json.dumps({"space_spec": asdict(MOCK_SPEC)}), encoding="utf-8")
    assert main(["import", "--db", db, "--manifest", str(tmp_path / "senza_record.json")]) == 1

    bad_memory = json.dumps(
        {"space_spec": asdict(MOCK_SPEC), "records": [{"memory": {"id": "x"}, "embedding": QUERY}]},
        ensure_ascii=False,
    )
    (tmp_path / "memoria_troncata.json").write_text(bad_memory, encoding="utf-8")
    assert main(["import", "--db", db, "--manifest", str(tmp_path / "memoria_troncata.json")]) == 1

    assert "MANIFEST_INVALIDO" in capsys.readouterr().err
    assert not Path(db).exists()


def test_import_rejects_mixed_source_hashes_per_file(tmp_path, capsys):
    db = str(tmp_path / "brain.db")
    first = make_record("/tmp/a.md", "k:a1", 0.9)
    second = MemoryRecord(
        replace(replace(first.memory, chunk_key="k:a2"), source_sha256="c" * 64), aligned_vector(0.4)
    )
    manifest = write_manifest(tmp_path / "manifest.json", MOCK_SPEC, [first, second])

    assert main(["import", "--db", db, "--manifest", str(manifest)]) == 1
    assert "MANIFEST_INVALIDO" in capsys.readouterr().err
    assert not Path(db).exists()
