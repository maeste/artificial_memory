"""Fixture condivise dei test storage (ST2, untracked_doc/02_AGENT_1_STORAGE.md).

Codice di test, non API di BrainStorage: record sintetici con vettori unitari
sugli assi, così le relazioni numeriche restano deterministiche.
"""

import hashlib
from pathlib import Path

import pytest

from src.contracts import MemoryRecord, MemoryView, SpaceSpec, make_memory_id
from src.storage.db import BrainStorage


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "brain.db")


@pytest.fixture
def storage(db_path):
    db = BrainStorage(db_path, SpaceSpec("mock", "semantic-hash-mock", "1"), "numpy")
    yield db
    db.close()


@pytest.fixture
def record_factory():
    def build(path, key, text, axis):
        path = str(Path(path).resolve())
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        content_hash = hashlib.sha256((key + chr(10) + text).encode()).hexdigest()
        view = MemoryView(
            id=make_memory_id(path, key, content_hash),
            file_path=path,
            modality="markdown",
            chunk_key=key,
            content_sha256=content_hash,
            source_sha256=source_hash,
            context_header=Path(path).name + " > " + key,
            snippet_text=text[:350],
            content_text=text,
            created_at=1.0,
        )
        vector = [0.0] * 256
        vector[axis] = 1.0
        return MemoryRecord(view, vector)

    return build
