"""Configurazione centralizzata di Ambient Second Brain (Orchestrator).

Unico punto che possiede path e valori di configurazione (contratti §2):
RuntimePaths, SpaceSpec, SearchSettings, ExportSettings e i default operativi.
Le opzioni CLI e il file salvato nell'export indicano i valori effettivi.
"""

import os
import tempfile
import uuid
from dataclasses import dataclass, field

from src.contracts import EMBED_DIM, RuntimePaths, SearchSettings, SpaceSpec

# Default operativi (contratti §2).
DEFAULT_ALPHA = 0.35
DEFAULT_SOURCE_TTL_SEC = 120
DEFAULT_COMMAND_TIMEOUT_MS = 2000
DEFAULT_EMBED_TIMEOUT_MS = 30000
DEFAULT_STARTUP_TIMEOUT_SEC = 300
DEFAULT_MAX_PINS = 12
DEFAULT_SESSION_LIMIT_SEC = 3600
DEFAULT_SESSION_MAX_INPUTS = 2000
DEFAULT_SESSION_AUDIO_MAX_BYTES = 125829120  # 120 MiB

# Prefissi/mock per lo spazio di sviluppo (contratti §2).
MOCK_MODEL_ID = 'semantic-hash-mock'
MOCK_REVISION = '1'
REAL_MODEL_ID = 'google/embeddinggemma-2'

RUNTIME_BASENAME = 'artificial-memory'


@dataclass(frozen=True)
class AppConfig:
    """Valori effettivi di una esecuzione: backend fisso per tutta la corsa."""

    space: SpaceSpec
    paths: RuntimePaths
    search: SearchSettings = field(default_factory=SearchSettings)
    device: str = 'auto'  # auto | cpu | cuda
    mic_enabled: bool = True
    alpha: float = DEFAULT_ALPHA
    source_ttl_sec: float = DEFAULT_SOURCE_TTL_SEC
    command_timeout_ms: int = DEFAULT_COMMAND_TIMEOUT_MS
    embed_timeout_ms: int = DEFAULT_EMBED_TIMEOUT_MS
    startup_timeout_sec: int = DEFAULT_STARTUP_TIMEOUT_SEC
    max_pins: int = DEFAULT_MAX_PINS
    session_limit_sec: float = DEFAULT_SESSION_LIMIT_SEC
    session_max_inputs: int = DEFAULT_SESSION_MAX_INPUTS
    session_audio_max_bytes: int = DEFAULT_SESSION_AUDIO_MAX_BYTES


def new_run_id() -> str:
    """UUID per la directory runtime; riusa lo stesso valore in caso di retry."""
    return uuid.uuid4().hex


def runtime_root(override: str | None = None) -> str:
    """Radice delle directory di esecuzione, permessi 0700, senza nomi globali fissi."""
    if override:
        root = os.path.abspath(override)
    elif os.environ.get('XDG_RUNTIME_DIR'):
        root = os.path.join(os.environ['XDG_RUNTIME_DIR'], RUNTIME_BASENAME)
    else:
        root = os.path.join(tempfile.gettempdir(), f'{RUNTIME_BASENAME}-{os.getpid()}')
    os.makedirs(root, mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def make_runtime_paths(
    run_id: str,
    db_path: str,
    watch_dir: str,
    export_dir: str,
    runtime_root_override: str | None = None,
) -> RuntimePaths:
    """Directory per esecuzione: socket, state.json e WAV live nel runtime."""
    run_dir = os.path.join(runtime_root(runtime_root_override), run_id)
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    os.chmod(run_dir, 0o700)
    return RuntimePaths(
        run_dir=run_dir,
        embed_endpoint=f'ipc://{os.path.join(run_dir, "embed.sock")}',
        events_endpoint=f'ipc://{os.path.join(run_dir, "events.sock")}',
        control_endpoint=f'ipc://{os.path.join(run_dir, "control.sock")}',
        db_path=os.path.abspath(db_path),
        watch_dir=os.path.abspath(watch_dir),
        export_dir=os.path.abspath(export_dir),
    )


def default_space(backend: str, revision: str) -> SpaceSpec:
    """Costruisce lo SpaceSpec per backend; la revisione reale è un commit HF risolto."""
    if backend == 'mock':
        return SpaceSpec(backend='mock', model_id=MOCK_MODEL_ID, revision=MOCK_REVISION, dimension=EMBED_DIM)
    if backend != 'real':
        raise ValueError(f'INVALID_REQUEST: backend non valido: {backend!r}')
    if not revision:
        raise ValueError('INVALID_REQUEST: revision reale obbligatoria (commit HF risolto, mai "main")')
    return SpaceSpec(backend='real', model_id=REAL_MODEL_ID, revision=revision, dimension=EMBED_DIM)


def default_data_layout(root: str) -> tuple[str, str, str]:
    """Layout persistente: DB per backend e directory export. Restituisce (db_real, db_mock, export_dir)."""
    data_dir = os.path.join(root, 'data')
    export_dir = os.path.join(root, 'exports')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    return (
        os.path.join(data_dir, 'brain-real.db'),
        os.path.join(data_dir, 'brain-mock.db'),
        export_dir,
    )
