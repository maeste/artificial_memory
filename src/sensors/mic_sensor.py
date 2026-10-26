"""Microfono — cattura, VAD RMS, segmentazione limitata e WAV univoci (piano L3).

PCM mono 16 kHz a blocchi da 30 ms/480 campioni int16. Prima sounddevice; se
il backend fallisce e ``parec`` è disponibile, subprocess con argomenti
``--rate=16000 --channels=1 --format=s16le``. Nessun fallback finto con file
di test: se nessun backend reale è disponibile, errore esplicito.

Il VAD è energetico (RMS): in ambienti rumorosi può tagliare o trascinare
code di parlato; il limite è documentato nel README e visibile inagnostica.
"""

import os
import shutil
import subprocess
import threading
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.contracts import AUDIO_SOURCES, LiveInput

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 480  # 30 ms
BLOCK_SEC = BLOCK_SAMPLES / SAMPLE_RATE
BYTES_PER_BLOCK = BLOCK_SAMPLES * 2  # int16 mono
CALIBRATION_SEC = 1.0
MIN_THRESHOLD = 0.008
NOISE_FACTOR = 3.0
NOISE_BETA = 0.1  # adattamento del pavimento solo su frame non vocali
MIN_SPEECH_SEC = 0.6
END_SILENCE_SEC = 0.6
MAX_SEGMENT_SEC = 12.0
OVERLAP_SEC = 0.3
PAREC_BASE_ARGS = ('parec', '--rate=16000', '--channels=1', '--format=s16le')
# ATTENZIONE: parec SENZA -d registra il MICROFONO. La cattura di sistema passa
# sempre un device esplicito (es. @DEFAULT_MONITOR@) via _parec_argv.


@dataclass
class AudioSegment:
    """Segmento completato: PCM int16 originale, durata totale e parlato."""

    samples: 'np.ndarray'
    duration_sec: float
    speech_sec: float


class VadSegmenter:
    """VAD energetico RMS su blocchi normalizzati, stato puro (nessun hardware).

    Primo secondo di calibrazione in silenzio; soglia iniziale
    ``max(min_threshold, noise_factor * noise_rms)``, configurabile. Il
    pavimento di rumore si aggiorna soltanto su frame non vocali.
    """

    def __init__(
        self,
        *,
        calibration_sec: float = CALIBRATION_SEC,
        min_threshold: float = MIN_THRESHOLD,
        noise_factor: float = NOISE_FACTOR,
        min_speech_sec: float = MIN_SPEECH_SEC,
        end_silence_sec: float = END_SILENCE_SEC,
        max_segment_sec: float = MAX_SEGMENT_SEC,
        overlap_sec: float = OVERLAP_SEC,
    ):
        self.min_threshold = float(min_threshold)
        self.noise_factor = float(noise_factor)
        self.min_speech_sec = float(min_speech_sec)
        self._calibration_blocks = max(1, round(calibration_sec / BLOCK_SEC))
        self._end_blocks = max(1, round(end_silence_sec / BLOCK_SEC))
        self._max_blocks = max(1, round(max_segment_sec / BLOCK_SEC))
        self._overlap_samples = round(overlap_sec * SAMPLE_RATE)
        self._calibration_rms: list[float] = []
        self._noise_rms: float | None = None
        self._threshold: float | None = None
        self._blocks: list['np.ndarray'] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._prefix: 'np.ndarray | None' = None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    @property
    def noise_rms(self) -> float | None:
        return self._noise_rms

    def feed(self, block: 'np.ndarray') -> list[AudioSegment]:
        """Assimila un blocco PCM int16 e restituisce i segmenti completati."""
        if block.ndim != 1 or block.size != BLOCK_SAMPLES:
            raise ValueError(f'INVALID_REQUEST: blocco da {BLOCK_SAMPLES} campioni, ricevuti {block.size}')
        float_block = block.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(float_block))))
        if self._threshold is None:
            self._calibration_rms.append(rms)
            if len(self._calibration_rms) >= self._calibration_blocks:
                self._noise_rms = float(np.mean(self._calibration_rms))
                self._threshold = self._computed_threshold()
            return []
        if rms >= self._threshold:
            self._append_speech(block)
            return self._check_emit()
        # frame non vocali: solo qui il pavimento di rumore si aggiorna
        self._noise_rms = (1.0 - NOISE_BETA) * self._noise_rms + NOISE_BETA * rms
        self._threshold = self._computed_threshold()
        if self._blocks:
            self._silence_frames += 1
            self._blocks.append(block)
            return self._check_emit()
        return []

    def flush(self) -> list[AudioSegment]:
        """Flush controllato (pausa): chiude il segmento pendente, se valido."""
        if not self._blocks:
            return []
        segment = self._finalize(forced=False)
        return [segment] if segment is not None else []

    def _computed_threshold(self) -> float:
        return max(self.min_threshold, self.noise_factor * float(self._noise_rms))

    def _append_speech(self, block: 'np.ndarray') -> None:
        if not self._blocks and self._prefix is not None:
            # overlap del segmento forzato precedente (piano L3)
            self._blocks.append(self._prefix)
            self._speech_frames += self._prefix.size // BLOCK_SAMPLES
            self._prefix = None
        self._silence_frames = 0
        self._speech_frames += 1
        self._blocks.append(block)

    def _check_emit(self) -> list[AudioSegment]:
        if not self._blocks:
            return []
        total_frames = sum(b.size for b in self._blocks) // BLOCK_SAMPLES
        if self._silence_frames >= self._end_blocks:
            segment = self._finalize(forced=False)
        elif total_frames >= self._max_blocks:
            segment = self._finalize(forced=True)
        else:
            return []
        return [segment] if segment is not None else []

    def _finalize(self, *, forced: bool) -> AudioSegment | None:
        samples = np.concatenate(self._blocks) if len(self._blocks) > 1 else self._blocks[0].copy()
        speech_sec = self._speech_frames * BLOCK_SEC
        self._blocks = []
        self._speech_frames = 0
        self._silence_frames = 0
        if forced:
            # il limite include l'audio inviato al modello; la coda parte con overlap
            self._prefix = samples[-self._overlap_samples :].copy()
        if speech_sec < self.min_speech_sec:
            return None  # segmento troppo breve: eliminato
        return AudioSegment(
            samples=samples,
            duration_sec=samples.size / SAMPLE_RATE,
            speech_sec=speech_sec,
        )


def write_wav(path: str, samples: 'np.ndarray') -> None:
    """Scrive un WAV mono 16 kHz int16 e chiude il file prima di restituire."""
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.astype('<i2').tobytes())


class MicSensor:
    """Producer audio: blocchi PCM → VAD → WAV univoci → LiveInput.

    Parametrizzato su ``source``/``file_prefix``: mic_audio ('mic-') o
    system_audio ('sysaudio-') — stessa pipeline VAD/WAV (F1). Non attende
    inferenza: l'output va nella coda limitata del runner (LiveInputQueue).
    La pausa chiude il segmento pendente con flush controllato; la sorgente
    testuale resta attiva.
    """

    def __init__(
        self,
        *,
        audio_root: str,
        session_provider: Callable[[], str],
        clock: Callable[[], float] | None = None,
        segmenter: VadSegmenter | None = None,
        source: str = 'mic_audio',
        file_prefix: str = 'mic',
    ):
        if source not in AUDIO_SOURCES:
            raise ValueError(
                f'INVALID_REQUEST: sorgente MicSensor non ammessa: {source!r} '
                '(mic_audio|system_audio)'
            )
        self._audio_root = audio_root
        self._session_provider = session_provider
        self._clock = clock if clock is not None else time.time
        self._segmenter = segmenter if segmenter is not None else VadSegmenter()
        self._paused = False
        self.source = source
        self.file_prefix = file_prefix

    def feed_block(self, block: 'np.ndarray') -> list[LiveInput]:
        """Un blocco da 30 ms: segmentazione e, per ogni segmento, WAV + input."""
        if self._paused:
            return []
        return [self._emit(segment) for segment in self._segmenter.feed(block)]

    def flush(self) -> list[LiveInput]:
        """Pausa mic: flush controllato del segmento pendente."""
        return [self._emit(segment) for segment in self._segmenter.flush()]

    def pause(self) -> list[LiveInput]:
        self._paused = True
        return self.flush()

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def _emit(self, segment: AudioSegment) -> LiveInput:
        session_id = self._session_provider()
        directory = os.path.join(self._audio_root, session_id)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'{self.file_prefix}-{uuid.uuid4().hex}.wav')
        write_wav(path, segment.samples)  # il file è chiuso prima di accodare
        return LiveInput(
            id=uuid.uuid4().hex,
            session_id=session_id,
            source=self.source,
            occurred_at=self._clock(),
            audio_path=path,
            duration_sec=segment.duration_sec,
        )


class AudioCapture:
    """Cattura reale PCM mono 16 kHz: sounddevice, poi parec come fallback.

    Con ``parec_device`` impostato (es. ``@DEFAULT_MONITOR@``) la cattura è
    SOLO parec con ``-d <parec_device>``: il ramo sounddevice non è tentato e
    il microfono non è mai registrato per errore (F1). Espone ``backend`` e
    ``device`` effettivi per la diagnostica. Nessun fallback finto: senza
    backend disponibile solleva RuntimeError.
    """

    def __init__(self, on_block: Callable[['np.ndarray'], None], *, device=None, parec_device=None):
        self.on_block = on_block
        self.device = device
        self.parec_device = parec_device
        self.backend: str | None = None
        self.device_label: str | None = None
        self._stream = None
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if self.parec_device is not None:
            if self._try_parec():
                return
            raise RuntimeError(
                f'AUDIO_UNAVAILABLE: parec non disponibile per la cattura di sistema '
                f'({self.parec_device})'
            )
        if self._try_sounddevice():
            return
        if self._try_parec():
            return
        raise RuntimeError(
            'MIC_UNAVAILABLE: nessun backend audio (sounddevice senza PortAudio, parec assente)'
        )

    def _parec_argv(self) -> tuple[str, ...]:
        if self.parec_device is not None:
            return PAREC_BASE_ARGS + ('-d', self.parec_device)
        return PAREC_BASE_ARGS

    def _try_sounddevice(self) -> bool:
        try:
            import sounddevice as sd
        except Exception:  # noqa: BLE001 — backend assente o PortAudio rotto
            return False
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SAMPLES,
                channels=1,
                dtype='int16',
                device=self.device,
            ) as stream:
                self.backend = 'sounddevice'
                target = self.device if self.device is not None else sd.default.device[1]
                self.device_label = str(sd.query_devices(target)) if target is not None else 'default'
                self._stream = stream
                while not self._stopping.is_set():
                    data, overflowed = stream.read(BLOCK_SAMPLES)
                    if overflowed:
                        continue
                    block = np.frombuffer(bytes(data), dtype='<i2')
                    if block.size == BLOCK_SAMPLES:
                        self.on_block(block)
            return True
        except Exception:  # noqa: BLE001 — il device può sparire o rifiutare l'apertura
            self.backend = None
            self.device_label = None
            return False

    def _try_parec(self) -> bool:
        argv = self._parec_argv()
        binary = shutil.which(argv[0])
        if binary is None:
            return False
        try:
            self._process = subprocess.Popen(  # noqa: S603 — argomenti fissi, senza shell
                argv, executable=binary, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except OSError:
            return False
        self.backend = 'parec'
        self.device_label = self.parec_device if self.parec_device is not None else 'default'
        self._thread = threading.Thread(target=self._read_parec, name='parec-reader', daemon=True)
        self._thread.start()
        return True

    def _read_parec(self) -> None:
        process = self._process
        stream = process.stdout
        while not self._stopping.is_set():
            chunk = stream.read(BYTES_PER_BLOCK)
            if not chunk:
                break
            if len(chunk) == BYTES_PER_BLOCK:
                self.on_block(np.frombuffer(chunk, dtype='<i2'))

    def stop(self) -> None:
        self._stopping.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._stream = None
