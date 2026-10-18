"""Harness di valutazione — riusa producer e worker reali senza GTK (piano L4).

Interfaccia consumata da scripts/evaluate.py::

    ReplayHarness(space=..., manifest_path=..., backend=...).run(
        queries=lista_manifest, source='agent_text'|'mic_audio', repeat=N)

Esito per query: ``{'query_id', 'relevant', 'results', 'error'}`` con results
come dict ScoredMemory. Testo con debounce reale; WAV a blocchi 30 ms con
pacing reale, 1 s di silenzio prima (calibrazione VAD) e 800 ms dopo (flush).
Il fine-input è l'ultimo campione originale, escluso il padding artificiale.
Reset fra query indipendenti; l'evento dell'ultimo segmento è l'esito. Non
chiama ingest dal producer né replica ranking/EMA: tutto passa dal runner.
"""

import argparse
import json
import math
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.contracts import HudCommand, LiveInput, RuntimePaths, SearchSettings, SpaceSpec
from src.sensors.agent_sensor import AgentLogTail
from src.sensors.mic_sensor import BLOCK_SAMPLES, BLOCK_SEC, MicSensor, SAMPLE_RATE
from src.sensors.runner import RunnerService

FLUSH_SILENCE_SEC = 0.8
CALIBRATION_SEC = 1.0


class _CharTokenizer:
    """Tokenizer deterministica a livello di carattere per il backend mock
    (stessa fixture dei test Ingestion, piano G1: 1 char = 1 token)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(chr(int(i)) for i in ids)


def _load_tokenizer(space: SpaceSpec):
    if space.backend == 'mock':
        return _CharTokenizer()
    from transformers import AutoTokenizer  # noqa: PLC0415 — import pesante solo per il reale

    return AutoTokenizer.from_pretrained(space.model_id, revision=space.revision)


class _Pacer:
    """Mantiene il pacing reale dei blocchi da 30 ms (piano L4)."""

    def __init__(self, step_sec: float):
        self._step = step_sec
        self._next = time.time() + step_sec

    def wait(self) -> None:
        delay = self._next - time.time()
        if delay > 0:
            time.sleep(delay)
        self._next += self._step


class ReplayHarness:
    """Mondo completo per una valutazione: indice, runner reale, producer reali."""

    def __init__(
        self,
        space: SpaceSpec,
        manifest_path: str,
        backend: str,
        *,
        config: dict | None = None,
        work_dir: str | None = None,
        device: str = 'auto',
    ):
        if backend not in ('real', 'mock'):
            raise ValueError(f'INVALID_REQUEST: backend non ammesso: {backend!r} (real|mock)')
        self.space = space
        self.backend = backend
        manifest_file = Path(manifest_path).resolve()
        self.base = manifest_file.parent
        self.manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        config = config or {}
        self.search = SearchSettings(**(config.get('search') or {}))
        self.embed_timeout_sec = float(config.get('embed_timeout_ms', 30000)) / 1000.0
        self._work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix='am-replay-'))
        run_dir = self._work / 'run'
        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ('exports',):
            (self._work / sub).mkdir(parents=True, exist_ok=True)
        self.paths = RuntimePaths(
            run_dir=str(run_dir),
            embed_endpoint=f'ipc://{run_dir}/embed.sock',
            events_endpoint=f'ipc://{run_dir}/events.sock',
            control_endpoint=f'ipc://{run_dir}/control.sock',
            db_path=str(self._work / f'brain-{backend}.db'),
            watch_dir=str(self.base),
            export_dir=str(self._work / 'exports'),
        )
        self._embedder = self._make_embedder(space, backend, device)
        self.tokenizer = _load_tokenizer(space)
        self.storage = self._make_storage(space)
        self.service = RunnerService(self.storage, self._embedder, self.paths, self.search, space=space)
        self._index_corpus()

    # ------------------------------------------------------------------ setup

    def _make_embedder(self, space: SpaceSpec, backend: str, device: str):
        from src.inference.model import EmbeddingGemmaBackend, SemanticHashMock

        if backend == 'mock':
            return SemanticHashMock(space)
        return EmbeddingGemmaBackend(space, device=device)

    def _make_storage(self, space: SpaceSpec):
        from src.storage.db import BrainStorage

        return BrainStorage(self.paths.db_path, space)

    def _index_corpus(self) -> None:
        from src.ingestion.watcher import index_file

        for entry in self.manifest['files']:
            path = self.base / entry['path']
            indexed = index_file(str(path), self.storage, self._embedder, self.tokenizer, str(self._work))
            print(f'replay: corpus {entry["id"]} {path.name}: {"indicizzato" if indexed else "invariato"}',
                  file=sys.stderr)

    def close(self) -> None:
        """Chiude storage; la directory di lavoro resta per ispezione post-mortem."""
        self.storage.close()

    # ------------------------------------------------------------------ run

    def run(self, queries: list[dict], source: str, repeat: int = 1) -> list[dict]:
        """Ripete ogni query indipendente; un esito per query e repeat."""
        if source not in ('agent_text', 'mic_audio'):
            raise ValueError(f'INVALID_REQUEST: sorgente non ammessa: {source!r}')
        if repeat < 1:
            raise ValueError(f'INVALID_REQUEST: repeat non valido: {repeat}')
        outcomes = []
        for query in queries:
            for rep in range(1, repeat + 1):
                outcomes.append(self._run_query(query, source, rep))
        return outcomes

    def _run_query(self, query: dict, source: str, rep: int) -> dict:
        query_id = query['id']
        timeout = self._query_timeout(query, source)
        started = time.time()
        self.last_segments = 1
        self.last_end_of_input = started
        self._new_session()
        try:
            if source == 'agent_text':
                state = self._replay_text(query, rep, timeout)
            else:
                state = self._replay_audio(query, rep, timeout)
        except Exception as exc:  # noqa: BLE001 — l'errore è l'esito, non un crash della valutazione
            print(f'replay: query {query_id} rep{rep} FALLITA: {exc}', file=sys.stderr)
            return {
                'query_id': query_id,
                'relevant': query.get('relevant', []),
                'results': [],
                'error': f'{type(exc).__name__}: {exc}',
            }
        elapsed = time.time() - started
        print(
            json.dumps({
                'query_id': query_id, 'repeat': rep, 'source': source,
                'segments': self.last_segments,
                'end_of_input': round(self.last_end_of_input, 3),
                'sequence': state.sequence, 'n_results': len(state.results),
                'elapsed_sec': round(elapsed, 3), 'timeout_sec': round(timeout, 3),
            }, ensure_ascii=False),
            file=sys.stderr,
        )
        return {
            'query_id': query_id,
            'relevant': query.get('relevant', []),
            'results': [asdict(item) for item in state.results],
            'error': None,
        }

    def _new_session(self) -> None:
        response = self.service.handle(
            HudCommand(command_id=f'replay-{uuid.uuid4().hex}', session_id=self.service.session_id,
                       action='new_session')
        )
        if not response.success:
            raise RuntimeError(f'reset fallito: {response.error_code}')

    def _replay_text(self, query: dict, rep: int, timeout: float):
        log = self._work / f'query-{query["id"]}-rep{rep}-agent.log'
        log.write_text(query['text'] + '\n', encoding='utf-8')
        tail = AgentLogTail(
            str(log), 'plain', start='beginning',
            session_provider=lambda: self.service.session_id,
        )
        deadline = time.time() + timeout
        while True:
            item = tail.poll()  # debounce reale del producer testuale
            if item is not None:
                break
            if time.time() > deadline:
                raise TimeoutError('TIMEOUT: nessun input testuale entro il timeout della query')
            time.sleep(0.02)
        return self.service.ingest(self._tag(item, query['id'], rep, 'agent_text', 0))

    def _replay_audio(self, query: dict, rep: int, timeout: float):
        wav = self.base / query['audio_path']
        samples = _read_wav_mono16k(wav)
        mic = MicSensor(audio_root=self.paths.run_dir, session_provider=lambda: self.service.session_id)
        silence = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
        pacer = _Pacer(BLOCK_SEC)
        inputs: list[LiveInput] = []
        # 1 s di silenzio: calibrazione del VAD prima del replay (piano L4)
        for _ in range(round(CALIBRATION_SEC / BLOCK_SEC)):
            pacer.wait()
            inputs.extend(mic.feed_block(silence))
        blocks = [samples[i : i + BLOCK_SAMPLES] for i in range(0, samples.size, BLOCK_SAMPLES)]
        if blocks and blocks[-1].size < BLOCK_SAMPLES:
            blocks[-1] = np.concatenate(
                [blocks[-1], np.zeros(BLOCK_SAMPLES - blocks[-1].size, dtype=np.int16)]
            )
        for block in blocks:
            pacer.wait()  # pacing reale, come la cattura dal microfono
            inputs.extend(mic.feed_block(block))
        end_of_input = time.time()  # l'ultimo campione originale, prima del padding di flush
        for _ in range(round(FLUSH_SILENCE_SEC / BLOCK_SEC)):
            pacer.wait()
            inputs.extend(mic.feed_block(silence))
        inputs.extend(mic.flush())
        if not inputs:
            raise ValueError('FAIL: nessun segmento prodotto dal replay audio')
        self.last_segments = len(inputs)
        self.last_end_of_input = end_of_input
        state = None
        for index, item in enumerate(inputs):
            tagged = self._tag(item, query['id'], rep, 'mic_audio', index)
            state = self.service.ingest(tagged)  # attesa dell'evento di ogni segmento
        return state

    def _tag(self, item: LiveInput, query_id: str, rep: int, source: str, index: int) -> LiveInput:
        return replace(item, id=f'{query_id}/rep{rep}/{source}/{index}')

    def _query_timeout(self, query: dict, source: str) -> float:
        # durata WAV + segmenti * (2*embed_timeout + 5) + 2 (piano L4)
        if source == 'agent_text':
            wav_sec, n_segments = 0.0, 1
        else:
            with wave.open(str(self.base / query['audio_path']), 'rb') as handle:
                wav_sec = handle.getnframes() / handle.getframerate()
            n_segments = max(1, math.ceil(wav_sec / 12.0))
        return wav_sec + n_segments * (2 * self.embed_timeout_sec + 5) + 2


def _read_wav_mono16k(path: Path) -> 'np.ndarray':
    with wave.open(str(path), 'rb') as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f'INVALID_MEDIA: WAV non mono 16 kHz int16: {path.name} '
                f'({handle.getnchannels()}ch, {handle.getframerate()}Hz)'
            )
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype='<i2')


def main(argv: list[str] | None = None) -> int:
    """uv run python -m src.sensors.replay --config PATH --manifest PATH ..."""
    from src.sensors.runner import load_config

    parser = argparse.ArgumentParser(description='Replay di valutazione sui producer reali (Agent-4)')
    parser.add_argument('--config', default=None, help='config JSON (space, search, timeout)')
    parser.add_argument('--manifest', default='tests/fixtures/ambient/manifest.json')
    parser.add_argument('--split', choices=('calibration', 'test'), required=True)
    parser.add_argument('--source', choices=('agent_text', 'mic_audio'), required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--work-dir', default=None, help='directory di lavoro (default: temp)')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    space_cfg = config.get('space')
    if space_cfg:
        space = SpaceSpec(**space_cfg)
    else:
        from src.inference.model import mock_space

        space = mock_space()
    harness = ReplayHarness(space, args.manifest, space.backend, config=config, work_dir=args.work_dir)
    queries = [q for q in harness.manifest['queries'] if q['split'] == args.split]
    if not queries:
        print(f'replay: split {args.split} vuoto', file=sys.stderr)
        return 1
    try:
        outcomes = harness.run(queries, source=args.source, repeat=args.repeat)
    finally:
        harness.close()
    failed = [outcome for outcome in outcomes if outcome['error']]
    print(json.dumps({'outcomes': outcomes, 'failed': len(failed), 'total': len(outcomes)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
