"""Genera l'export di esempio per la consegna (Orchestrator, O5).

Usa assemble_and_export_context reale (HUD H2) con memorie vere da
data/brain-real.db, pin con snapshot e input di sessione dimostrativi,
trascrizione off (default del PoC). Prova concreta di S7: Markdown con
riferimenti risolvibili, allegati verificati per digest, manifest.
"""

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import (  # noqa: E402
    ContextSnapshot,
    ExportSettings,
    LiveInput,
    LiveStateEvent,
    PinnedMemory,
    SpaceSpec,
)
from src.hud.context_exporter import assemble_and_export_context  # noqa: E402
from src.storage.db import BrainStorage  # noqa: E402

SESSION_ID = 'demo-' + uuid.uuid4().hex[:8]
FIXTURES = Path('tests/fixtures/ambient').resolve()
CREATED = 1792300000.0


def pick(storage: BrainStorage, suffix: str, chunk_hint: str = ''):
    for record in storage.all_memories():
        if record.memory.file_path.endswith(suffix) and chunk_hint in record.memory.chunk_key:
            return record.memory
    raise SystemExit(f'memoria non trovata: {suffix} {chunk_hint}')


def main() -> int:
    space = SpaceSpec(**json.loads(Path('data/probe-config.json').read_text())['space'])
    storage = BrainStorage('data/brain-real.db', space)
    try:
        m1 = pick(storage, 'f01.md', 'Cache')
        m2 = pick(storage, 'f09.png')
        m3 = pick(storage, 'f13.mp4', 'video:40000:85000')
    finally:
        storage.close()

    inputs = [
        LiveInput(id='in-1', session_id=SESSION_ID, source='agent_text',
                  occurred_at=CREATED, text='Stiamo ragionando sulla cache del servizio.'),
        LiveInput(id='in-2', session_id=SESSION_ID, source='agent_text',
                  occurred_at=CREATED + 20, text='Quanto durano i risultati e quando li invalidiamo?'),
        LiveInput(id='in-3', session_id=SESSION_ID, source='mic_audio',
                  occurred_at=CREATED + 40, audio_path=str(FIXTURES / 'query-T1.wav'),
                  duration_sec=4.0),
    ]
    snapshot = ContextSnapshot(id='snap-demo', session_id=SESSION_ID,
                               captured_at=CREATED + 45, input_ids=['in-1', 'in-2', 'in-3'])
    pinned = [
        PinnedMemory(memory=m1, pinned_at=CREATED + 46, snapshot=snapshot),
        PinnedMemory(memory=m2, pinned_at=CREATED + 47, snapshot=snapshot),
        PinnedMemory(memory=m3, pinned_at=CREATED + 48, snapshot=snapshot),
    ]
    state = LiveStateEvent(
        event_id='demo-event', sequence=1, session_id=SESSION_ID, space_id=space.space_id,
        backend='real', emitted_at=CREATED + 49, reason='pin',
        latest_inputs=inputs[-3:], results=[], pinned=pinned,
        sensor_status={'agent_text': 'active', 'mic_audio': 'suspended'}, dropped_inputs=0,
    )
    result = assemble_and_export_context(
        state=state, session_inputs=inputs, export_root='exports',
        settings=ExportSettings(
            transcription=sys.argv[1] if len(sys.argv) > 1 else 'off',
            asr_model_dir='./data/models/asr' if len(sys.argv) > 1 else None,
        ),
    )
    print(json.dumps({
        'export_dir': result.export_dir, 'markdown': result.markdown_path,
        'manifest': result.manifest_path, 'clipboard_status': result.clipboard_status,
        'warnings': result.warnings,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
