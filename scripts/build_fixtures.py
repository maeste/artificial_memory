"""Generatore del corpus di valutazione — Ambient Second Brain (08_EVALUATION §2).

Crea in --output: f01.md…f16.mp4 con nomi neutri, query-*.wav italiane
sintetiche, manifest.json normativo (con checksum) e document_requests.jsonl
per il probe di Fase 1 (chunk pronti senza estrattori applicativi).

I contenuti sono dati fittizi di test. Nessuna descrizione della tabella di
08 finisce nelle caption: il modello deve leggere pixel e audio.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import EmbedRequest, MemoryView, make_memory_id  # noqa: E402

GENERATOR_VERSION = '1.1'
FIXED_EPOCH = 1792300000.0
# Voci TTS distinte per query e narrazioni video: la run 1 del gate Fase 1 ha
# mostrato che una voce unica crea un confound audio-audio (la query vocale
# matcha le narrazioni per timbro prima che per contenuto). I 16 file del
# corpus non cambiano; cambia solo il timbro delle query WAV.
NARRATION_VOICE = 'it'
QUERY_VOICE = 'it+f4'
FIXTURE_NOTES = [
    'v1.1 (post gate-run-1): query WAV generate con voce espeak-ng "it+f4" '
    'mentre le narrazioni video restano "it", per rimuovere il confound '
    'voce-query == voce-narrazione osservato nel report fallito della prima '
    'esecuzione del gate Fase 1 (audio 2/4: C1/C2 mancate con cluster vocale '
    'piatto 0.66-0.73). Contenuti dei file, testi delle query e gold invariati.',
]
FPS = 2  # frame per secondo generati per i video
SIZE = (1200, 800)
FONT_CANDIDATES = [
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
]

CREATED_AT = FIXED_EPOCH

# --------------------------------------------------------------------------- contenuti fittizi

MARKDOWN_FILES = {
    'M1': (
        'f01.md',
        [
            ('Cache', 'La cache conserva i risultati delle interrogazioni per sessanta secondi dalla lettura. '
                      'Ogni scrittura sul database invalida subito le chiavi coinvolte, così la lettura successiva '
                      'ricalcola il valore aggiornato.'),
            ('Note operative', 'Lo strato di cache è trasparente per il chiamante e non modifica la semantica delle '
                               'risposte.'),
        ],
    ),
    'M2': (
        'f02.md',
        [
            ('Backup', 'La copia di backup viene eseguita ogni giorno alle due di notte. Con questa cadenza la '
                       'perdita massima di dati accettata in caso di guasto è di ventiquattro ore di scritture.'),
            ('Verifiche', 'Il ripristino viene provato monthly su ambiente separato.'),
        ],
    ),
    'M3': (
        'f03.md',
        [
            ('Autenticazione', 'L’access token scade dopo quindici minuti dalla emissione. Il client rinnova la '
                              'sessione tramite il refresh token, che resta valido per giorni e può essere revocato.'),
        ],
    ),
    'M4': (
        'f04.md',
        [
            ('Pane fatto in casa', 'Impastare cinquecento grammi di farina con acqua tiepida e lievito fresco. '
                                   'Lasciare lievitare la massa coperta per due ore prima di infornare.'),
        ],
    ),
}

PDF_FILES = {
    'P1': (
        'f05.pdf',
        [
            'Introduzione\n\nQuesto documento descrive la strategia di rilascio progressivo del servizio.',
            'Rilascio progressivo\n\nIl rollout canary avvia al cinque per cento del traffico complessivo. '
            'Gli errori vengono monitorati in tempo reale: il rilascio si interrompe con rollback automatico '
            'quando la percentuale di errori supera il due per cento.',
            'Appendice A\n\nNote operative e cronologia delle revisioni del documento.',
        ],
    ),
    'P2': (
        'f06.pdf',
        [
            'Indice\n\n1. Glossario\n2. Schema outbox',
            'Glossario\n\nCanale: coda di pubblicazione dei messaggi. Idempotenza: proprietà di elaborazione '
            'senza duplicati.',
            'Schema outbox\n\nIl record outbox viene salvato nella stessa transazione del database e pubblicato '
            'solo dopo il commit, così un messaggio non viene mai emesso per una scrittura annullata.',
        ],
    ),
    'P3': (
        'f07.pdf',
        [
            'Introduzione\n\nSintesi delle misure di risposta del servizio.',
            '__CHART_LATENCY__',
        ],
    ),
    'P4': (
        'f08.pdf',
        [
            'Irrigazione\n\nL’irrigazione a goccia riduce i consumi idrici del giardino e mantiene il terriccio '
            'uniformemente umido.',
            'Terriccio\n            \nIl terriccio argilloso trattiene l’acqua più a lungo di quello sabbioso.',
        ],
    ),
}

VIDEO_SPECS = {
    'V1': {
        'file': 'f13.mp4', 'duration': 95.0, 'voice': None,
        'topic': 'cpu-then-traffic-then-end',
    },
    'V2': {
        'file': 'f14.mp4', 'duration': 45.0,
        'voice': 'Il ritiro di TLS uno punto zero è stato deciso: dal trenta novembre non sarà più supportato.',
        'topic': 'neutral-a',
    },
    'V3': {
        'file': 'f15.mp4', 'duration': 45.0,
        'voice': 'Il timeout dell’endpoint è fissato a due secondi per tutte le chiamate in uscita.',
        'topic': 'neutral-b',
    },
    'V4': {
        'file': 'f16.mp4', 'duration': 45.0,
        'voice': 'Per la pasta fresca impastiamo a mano la farina con le uova, poi l’impasto riposa mezz’ora.',
        'topic': 'neutral-c',
    },
}

QUERIES = [
    ('C1', 'calibration', 'Come si rinnova l’accesso dopo la scadenza del token?', [('M1' if False else 'M3', None)]),
    ('C2', 'calibration', 'Mostrami il grafico con il limite di latenza a cento millisecondi.', [('P3', 2)]),
    ('C3', 'calibration', 'Cerco il simbolo rosso di avvertimento con un punto esclamativo.', [('I3', None)]),
    ('C4', 'calibration', 'Dove abbiamo fissato il timeout del servizio a due secondi?', [('V3', None)]),
    ('T1', 'test', 'Quanto durano i risultati nella cache e quando li invalidiamo?', [('M1', None)]),
    ('T2', 'test', 'Qual è la perdita massima di dati accettata con il backup giornaliero?', [('M2', None)]),
    ('T3', 'test', 'Con quale percentuale di errori si interrompe il rilascio canary?', [('P1', 2)]),
    ('T4', 'test', 'Come pubblicare il messaggio soltanto dopo il commit della transazione?', [('P2', 3)]),
    ('T5', 'test', 'Cerco lo schema con produttore, coda FIFO e consumatore.', [('I1', None)]),
    ('T6', 'test', 'Mostrami il grafico della memoria che continua a salire per cinquanta minuti.', [('I2', None)]),
    ('T7', 'test', 'Nel filmato, quando passa il traffico dall’ambiente blu a quello verde?', [('V1', 'interval')]),
    ('T8', 'test', 'Da quale data abbiamo deciso di ritirare TLS uno punto zero?', [('V2', None)]),
]

NEGATIVES = [
    ('N1', 'calibration', 'orari dei treni per Venezia'),
    ('N2', 'calibration', 'fasi lunari'),
    ('N3', 'calibration', 'istruzioni per una chitarra'),
    ('N4', 'calibration', 'regole degli scacchi'),
    ('N5', 'calibration', 'itinerario sulle Dolomiti'),
    ('N6', 'test', 'risultati del campionato di basket'),
    ('N7', 'test', 'visita a un museo egizio'),
    ('N8', 'test', 'accordi di un pianoforte'),
    ('N9', 'test', 'migrazione delle balene'),
    ('N10', 'test', 'storia della ceramica'),
]

V1_GOLD_INTERVAL = (40.0, 85.0)


# --------------------------------------------------------------------------- helper


def _font(size: int):
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _run(cmd: list[str], timeout: int = 120) -> None:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f'comando fallito {cmd[:3]}: {result.stderr.decode(errors="replace")[:300]}')


def _espeak_wav(text: str, out: Path, voice: str) -> None:
    _run(['espeak-ng', '-v', voice, '-s', '150', '-w', str(out), text])


def _wav_to_16k_mono(src: Path, dst: Path) -> None:
    _run(['ffmpeg', '-loglevel', 'error', '-y', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-i', str(src), '-ar', '16000', '-ac', '1', str(dst)])


def _deterministic_pdf_id(path: Path) -> None:
    """Sostituisce il /ID casuale del trailer con uno deterministico (fixture riproducibili)."""
    import re as _re

    stable = hashlib.sha256(f'am-fixture/{path.name}'.encode('utf-8')).hexdigest()[:32].upper().encode()
    data = path.read_bytes()
    patched = _re.sub(rb'/ID\[<[0-9A-F]+><[0-9A-F]+>\]', b'/ID[<' + stable + b'><' + stable + b'>]', data)
    path.write_bytes(patched)


def _repeat_voice(phrase: str, duration_sec: float, out_16k: Path, workdir: Path) -> None:
    """Ripete la frase (voce narrazione) con pause fino alla durata richiesta."""
    one = workdir / 'phrase.wav'
    _espeak_wav(phrase, one, NARRATION_VOICE)
    with wave.open(str(one), 'rb') as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    rate, width, channels = params.framerate, params.sampwidth, params.nchannels
    phrase_duration = params.nframes / rate
    silence = b'\x00' * int(rate * 1.1) * width * channels
    chunks, total = [], 0.0
    while total < duration_sec:
        chunks.append(frames)
        chunks.append(silence)
        total += phrase_duration + 1.1
    long_wav = workdir / 'voice-long.wav'
    with wave.open(str(long_wav), 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        for chunk in chunks:
            handle.writeframes(chunk)
    _wav_to_16k_mono(long_wav, out_16k)


# --------------------------------------------------------------------------- immagini per video


def _new_canvas(color=(245, 245, 245)):
    from PIL import Image, ImageDraw

    image = Image.new('RGB', SIZE, color)
    return image, ImageDraw.Draw(image)


def _video_frame(topic: str, t: float, duration: float):
    """Un frame del video `topic` all'istante t (secondi)."""
    from PIL import Image, ImageDraw

    image = Image.new('RGB', SIZE, (18, 18, 20))
    draw = ImageDraw.Draw(image)
    font_big = _font(44)
    font = _font(28)

    if topic == 'cpu-then-traffic-then-end':
        if t < 40.0:
            draw.text((60, 40), 'Utilizzo CPU', fill=(240, 240, 240), font=font_big)
            draw.line((80, 700, 1140, 700), fill=(120, 120, 120), width=2)
            draw.line((80, 120, 80, 700), fill=(120, 120, 120), width=2)
            points = []
            for step in range(60):
                x = 80 + step * (1060 / 59)
                phase = t / 40.0
                value = 0.45 + 0.35 * math.sin(step * 0.35 + phase * 5.0)
                points.append((x, 700 - value * 520))
            draw.line(points, fill=(90, 170, 255), width=4)
        elif t < 85.0:
            progress = (t - 40.0) / 45.0
            blue = (40, 90, 200)
            green = (40, 170, 80)
            color = tuple(int(b + (g - b) * progress) for b, g in zip(blue, green))
            draw.rectangle((100, 100, 1100, 660), fill=color)
            for dot in range(14):
                angle = 0.9 * t + dot * 0.7
                x = 160 + (dot * 70) + 30 * math.sin(angle)
                y = 200 + 200 * (0.5 + 0.5 * math.sin(angle * 1.3 + dot))
                draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(240, 240, 240))
            draw.text((60, 40), 'Sessione di prova', fill=(240, 240, 240), font=font_big)
        else:
            draw.rectangle((0, 0, SIZE[0], SIZE[1]), fill=(24, 24, 26))
            draw.text((430, 360), 'Fine', fill=(230, 230, 230), font=font_big)
    else:
        # Slide neutre per i video parlati: geometrie astratte, nessun testo decisionale.
        variants = {
            'neutral-a': [(255, 120, 90), (120, 160, 255)],
            'neutral-b': [(140, 220, 160), (230, 200, 110)],
            'neutral-c': [(210, 150, 220), (150, 210, 230)],
        }
        primary, secondary = variants[topic]
        draw.text((70, 50), 'Registro della sessione', fill=(235, 235, 235), font=font_big)
        for index in range(6):
            angle = 0.4 * t + index * 1.05
            cx = 200 + index * 160 + 26 * math.sin(angle)
            cy = 400 + 120 * math.sin(angle * 0.7 + index)
            radius = 46 + 14 * math.cos(angle)
            color = primary if index % 2 == 0 else secondary
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=6)
        draw.line((80, 660, 1120, 660), fill=(110, 110, 110), width=2)
        draw.text((80, 680), f'parte {int(t) // 15 + 1}', fill=(180, 180, 180), font=font)
    return image


def _still_image(spec: str):
    """Fixture immagine (I1–I4)."""
    from PIL import Image, ImageDraw

    image = Image.new('RGB', SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_big = _font(40)
    font = _font(30)

    if spec == 'I1':
        draw.text((400, 40), 'Flusso di elaborazione', fill=(20, 20, 20), font=font_big)
        boxes = [(80, 'Producer'), (470, 'FIFO Queue'), (860, 'Consumer')]
        for x, label in boxes:
            draw.rounded_rectangle((x, 300, x + 260, 470), radius=18, fill=(235, 240, 250),
                                   outline=(30, 30, 30), width=3)
            draw.text((x + 130 - len(label) * 8, 370), label, fill=(20, 20, 20), font=font)
        for x0, x1 in ((340, 470), (730, 860)):
            draw.line((x0, 385, x1, 385), fill=(20, 20, 20), width=5)
            draw.polygon([(x1, 385), (x1 - 22, 373), (x1 - 22, 397)], fill=(20, 20, 20))
    elif spec == 'I2':
        draw.text((360, 30), 'Memory leak', fill=(20, 20, 20), font=font_big)
        draw.line((90, 700, 1140, 700), fill=(60, 60, 60), width=2)
        draw.line((90, 120, 90, 700), fill=(60, 60, 60), width=2)
        points = [(90 + step * (1050 / 50), 660 - step * 10.4) for step in range(51)]
        draw.line(points, fill=(200, 40, 40), width=5)
        draw.text((1000, 690), '50 min', fill=(40, 40, 40), font=font)
        draw.text((30, 118), 'MB', fill=(40, 40, 40), font=font)
    elif spec == 'I3':
        draw.polygon([(600, 90), (950, 690), (250, 690)], fill=(210, 30, 30))
        text = '!'
        font_excl = _font(260)
        bbox = draw.textbbox((0, 0), text, font=font_excl)
        draw.text((600 - (bbox[2] - bbox[0]) / 2 - bbox[0], 250), text, fill=(255, 255, 255), font=font_excl)
    elif spec == 'I4':
        draw.text((420, 30), 'Fiore esaminato', fill=(20, 20, 20), font=font_big)
        for angle in range(0, 360, 45):
            x = 600 + 170 * math.cos(math.radians(angle))
            y = 300 + 130 * math.sin(math.radians(angle))
            draw.ellipse((x - 80, y - 60, x + 80, y + 60), outline=(170, 60, 150), width=4)
        draw.ellipse((540, 240, 660, 360), fill=(240, 200, 80), outline=(120, 90, 20), width=3)
        draw.line((600, 360, 600, 700), fill=(60, 130, 60), width=8)
        draw.ellipse((430, 480, 570, 560), fill=(90, 170, 90))
        draw.ellipse((630, 540, 770, 620), fill=(90, 170, 90))
        draw.text((690, 270), 'petalo', fill=(20, 20, 20), font=font)
        draw.text((620, 650), 'stelo', fill=(20, 20, 20), font=font)
    return image


# --------------------------------------------------------------------------- builder


class FixtureBuilder:
    def __init__(self, output: Path):
        self.output = output
        self.staging = output / '_staging'
        self.output.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, dict] = {}
        self.document_requests: list[dict] = []

    # -- markdown -----------------------------------------------------------

    def build_markdown(self) -> None:
        for file_id, (name, sections) in MARKDOWN_FILES.items():
            path = self.output / name
            lines = []
            for title, body in sections:
                lines.append(f'# {title}\n\n{body}\n')
            path.write_text(''.join(lines), encoding='utf-8')
            self._register(file_id, path)
            for occurrence, (title, body) in enumerate(sections):
                self._add_text_chunk(
                    file_id=file_id, path=path, modality='markdown',
                    chunk_key=f'section:{title}:0:part:0',
                    logical_header=title, body=body, display_header=f'{name} > {title}',
                )

    # -- pdf ----------------------------------------------------------------

    def build_pdfs(self) -> None:
        import pymupdf as fitz

        for file_id, (name, pages) in PDF_FILES.items():
            path = self.output / name
            doc = fitz.open()
            for page_text in pages:
                page = doc.new_page(width=842, height=595)
                if page_text == '__CHART_LATENCY__':
                    self._draw_latency_chart(page)
                else:
                    page.insert_font(fontname='helv', fontfile=None)
                    cursor = fitz.Point(60, 70)
                    for paragraph in page_text.split('\n\n'):
                        cursor = page.insert_textbox(
                            fitz.Rect(50, cursor.y, 792, 545), paragraph, fontname='helv',
                            fontsize=12, lineheight=1.4,
                        ) and cursor + (0, 0)
                        cursor = fitz.Point(60, min(cursor.y + 30, 545))
            doc.save(str(path))
            doc.close()
            _deterministic_pdf_id(path)
            self._register(file_id, path)
            for index, page_text in enumerate(pages, start=1):
                chunk_key = f'page:{index}:part:0'
                display_header = f'{name} [Pag. {index}]'
                logical_header = f'[Pag. {index}]'
                if page_text == '__CHART_LATENCY__':
                    png = self._render_pdf_page(path, index)
                    body = 'Latenza (ms), soglia operativa 100 ms.'
                    self._add_media_hybrid_chunk(
                        file_id, path, 'pdf', chunk_key, display_header, body, png, page_num=index,
                    )
                else:
                    self._add_text_chunk(
                        file_id=file_id, path=path, modality='pdf', chunk_key=chunk_key,
                        logical_header=logical_header, body=page_text.replace('\n\n', '\n'),
                        display_header=display_header, page_num=index,
                    )

    def _draw_latency_chart(self, page) -> None:
        import pymupdf as fitz

        shape = page.new_shape()
        shape.draw_line(fitz.Point(60, 500), fitz.Point(780, 500))
        shape.draw_line(fitz.Point(60, 80), fitz.Point(60, 500))
        shape.finish(color=(0.2, 0.2, 0.2), width=1)
        shape.commit()
        points = []
        for step in range(40):
            x = 60 + step * 18
            y = 300 - 130 * math.sin(step * 0.33) - 40
            points.append(fitz.Point(x, y))
        shape = page.new_shape()
        shape.draw_polyline(points)
        shape.finish(color=(0.1, 0.35, 0.8), width=2)
        shape.commit()
        shape = page.new_shape()
        shape.draw_line(fitz.Point(60, 260), fitz.Point(780, 260))
        shape.finish(color=(0.8, 0.1, 0.1), width=1.5, dashes='[6 3] 0')
        shape.commit()
        page.insert_text(fitz.Point(640, 250), 'soglia 100 ms', fontname='helv', fontsize=13,
                         color=(0.75, 0.05, 0.05))
        page.insert_text(fitz.Point(300, 60), 'Latenza per richiesta', fontname='helv', fontsize=16)
        page.insert_text(fitz.Point(30, 262), '100', fontname='helv', fontsize=10)

    def _render_pdf_page(self, pdf_path: Path, page_num: int) -> Path:
        import pymupdf as fitz

        target_dir = self.staging / pdf_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        png = target_dir / f'page{page_num}.png'
        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=120)
        pix.save(str(png))
        doc.close()
        return png

    # -- immagini -----------------------------------------------------------

    def build_images(self) -> None:
        names = {'I1': 'f09.png', 'I2': 'f10.jpg', 'I3': 'f11.png', 'I4': 'f12.jpg'}
        for file_id, name in names.items():
            path = self.output / name
            image = _still_image(file_id)
            if name.endswith('.jpg'):
                image.save(str(path), quality=92)
            else:
                image.save(str(path))
            self._register(file_id, path)
            with path.open('rb') as handle:
                from PIL import Image

                with Image.open(handle) as opened:
                    width, height = opened.size
            self._add_media_chunk(
                file_id=file_id, path=path, modality='image', chunk_key='image:0',
                display_header=f'{name} > immagine',
                snippet=f'Immagine {width}x{height}',
            )

    # -- video --------------------------------------------------------------

    def build_videos(self, workdir: Path) -> None:
        for file_id, spec in VIDEO_SPECS.items():
            frames_dir = self.staging / spec['file'].replace('.mp4', '') / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            total_frames = int(spec['duration'] * FPS)
            for index in range(total_frames):
                t = index / FPS
                frame_path = frames_dir / f'frame_{index:05d}.png'
                _video_frame(spec['topic'], t, spec['duration']).save(str(frame_path))
                frame_paths.append(frame_path)
            concat = frames_dir / 'concat.txt'
            concat.write_text(
                ''.join(f"file '{frame_path.name}'\nduration {1 / FPS}\n" for frame_path in frame_paths)
                + f"file '{frame_paths[-1].name}'\n",
                encoding='utf-8',
            )
            video_only = workdir / f'{file_id}-video.mp4'
            _run(['ffmpeg', '-loglevel', 'error', '-y', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-f', 'concat', '-safe', '0',
                  '-i', str(concat), '-vsync', 'vfr', '-pix_fmt', 'yuv420p', str(video_only)])
            final = self.output / spec['file']
            if spec['voice']:
                audio_16k = self.staging / spec['file'].replace('.mp4', '') / 'audio_full_16k.wav'
                audio_16k.parent.mkdir(parents=True, exist_ok=True)
                _repeat_voice(spec['voice'], spec['duration'], audio_16k, workdir)
                _run(['ffmpeg', '-loglevel', 'error', '-y', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-i', str(video_only), '-i', str(audio_16k),
                      '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(final)])
            else:
                _run(['ffmpeg', '-loglevel', 'error', '-y', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-i', str(video_only),
                      '-c:v', 'copy', '-an', str(final)])
            self._register(file_id, final)
            self._add_video_chunks(file_id, final, spec, frame_paths)

    def _add_video_chunks(self, file_id: Path, path: Path, spec: dict, frame_paths: list[Path]) -> None:
        duration = spec['duration']
        windows = []
        start = 0.0
        while start < duration:
            end = min(start + 45.0, duration)
            windows.append((start, end))
            if end >= duration:
                break
            start += 40.0
        stem = spec['file'].replace('.mp4', '')
        for start, end in windows:
            chunk_key = f'video:{int(start * 1000)}:{int(end * 1000)}'
            picks = []
            for fraction in (0.1, 0.5, 0.9):
                t = start + (end - start) * fraction
                picks.append(frame_paths[min(int(t * FPS), len(frame_paths) - 1)])
            audio_path = None
            if spec['voice']:
                audio_path = self.staging / stem / f'window_{int(start)}_{int(end)}.wav'
                _run(['ffmpeg', '-loglevel', 'error', '-y', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-ss', f'{start:.3f}', '-t', f'{end - start:.3f}',
                      '-i', str(self.staging / stem / 'audio_full_16k.wav'), '-ar', '16000', '-ac', '1',
                      str(audio_path)])
            base = self.staging / stem
            header = f'{spec["file"]} > [{start:.1f}s, {end:.1f}s)'
            view, request = self._media_record(
                file_id=file_id, path=path, modality='video', chunk_key=chunk_key,
                display_header=header, snippet=f'Video {duration:.0f} s, finestra [{start:.0f}, {end:.0f})',
                image_paths=picks, audio_path=audio_path, start_sec=start, end_sec=end,
            )
            self.document_requests.append(
                {'file_id': file_id, 'memory': _view_dict(view), 'request': _request_dict(request)}
            )

    # -- record comuni ------------------------------------------------------

    def _register(self, file_id: str, path: Path) -> None:
        self.files[file_id] = {'id': file_id, 'path': path.name, 'sha256': _sha256_file(path)}

    def _source_hash(self, path: Path) -> str:
        return self.files[[k for k, v in self.files.items() if v['path'] == path.name][0]]['sha256']

    def _add_text_chunk(self, file_id: str, path: Path, modality: str, chunk_key: str,
                        logical_header: str, body: str, display_header: str, page_num: int | None = None) -> None:
        normalized = body.replace('\r\n', '\n')
        content_sha = _sha256_text(logical_header + '\n' + normalized)
        view = MemoryView(
            id=make_memory_id(str(path.resolve()), chunk_key, content_sha),
            file_path=str(path.resolve()), modality=modality, chunk_key=chunk_key,
            content_sha256=content_sha, source_sha256=self._source_hash(path),
            context_header=display_header, snippet_text=normalized[:350], content_text=normalized,
            page_num=page_num, created_at=CREATED_AT,
        )
        request = EmbedRequest(
            request_id=f'doc:{file_id}:{chunk_key}', is_query=False,
            text_payload=normalized, context_header=display_header,
        )
        self.document_requests.append(
            {'file_id': file_id, 'memory': _view_dict(view), 'request': _request_dict(request)}
        )

    def _add_media_chunk(self, file_id: str, path: Path, modality: str, chunk_key: str,
                         display_header: str, snippet: str) -> None:
        view, request = self._media_record(
            file_id=file_id, path=path, modality=modality, chunk_key=chunk_key,
            display_header=display_header, snippet=snippet, image_paths=[str(path.resolve())],
        )
        self.document_requests.append(
            {'file_id': file_id, 'memory': _view_dict(view), 'request': _request_dict(request)}
        )

    def _add_media_hybrid_chunk(self, file_id: str, path: Path, modality: str, chunk_key: str,
                                display_header: str, body: str, image: Path, page_num: int) -> None:
        view, request = self._media_record(
            file_id=file_id, path=path, modality=modality, chunk_key=chunk_key,
            display_header=display_header, snippet=f'Pagina {page_num}, contenuto visuale; allegato richiesto',
            image_paths=[str(image.resolve())], text=body, page_num=page_num,
        )
        self.document_requests.append(
            {'file_id': file_id, 'memory': _view_dict(view), 'request': _request_dict(request)}
        )

    def _media_record(self, file_id: str, path: Path, modality: str, chunk_key: str, display_header: str,
                      snippet: str, image_paths: list[str], audio_path: Path | None = None,
                      text: str = '', page_num: int | None = None, start_sec: float | None = None,
                      end_sec: float | None = None):
        source_sha = self._source_hash(path)
        content_sha = hashlib.sha256((source_sha + chr(0) + chunk_key).encode('utf-8')).hexdigest()
        view = MemoryView(
            id=make_memory_id(str(path.resolve()), chunk_key, content_sha),
            file_path=str(path.resolve()), modality=modality, chunk_key=chunk_key,
            content_sha256=content_sha, source_sha256=source_sha,
            context_header=display_header, snippet_text=snippet[:350], content_text=text,
            page_num=page_num, start_sec=start_sec, end_sec=end_sec, created_at=CREATED_AT,
        )
        request = EmbedRequest(
            request_id=f'doc:{file_id}:{chunk_key}', is_query=False,
            text_payload=text, context_header=display_header,
            image_paths=[str(Path(item).resolve()) for item in image_paths],
            audio_path=str(Path(audio_path).resolve()) if audio_path else None,
        )
        return view, request

    # -- query --------------------------------------------------------------

    def build_queries(self, workdir: Path) -> list[dict]:
        entries = []
        for query_id, split, text, gold in QUERIES:
            entries.append(self._one_query(query_id, split, text, gold, workdir))
        for query_id, split, text in NEGATIVES:
            entries.append(self._one_query(query_id, split, text, [], workdir))
        return entries

    def _one_query(self, query_id: str, split: str, text: str, gold: list, workdir: Path) -> dict:
        raw = workdir / f'query-{query_id}-raw.wav'
        final = self.output / f'query-{query_id}.wav'
        _espeak_wav(text, raw, QUERY_VOICE)
        _wav_to_16k_mono(raw, final)
        relevant = []
        for file_id, page in gold:
            if file_id == 'V1':
                relevant.append(
                    {'file_id': 'V1', 'start_sec': V1_GOLD_INTERVAL[0], 'end_sec': V1_GOLD_INTERVAL[1]}
                )
            elif page is not None:
                relevant.append({'file_id': file_id, 'page_num': page})
            else:
                relevant.append({'file_id': file_id})
        return {
            'id': query_id, 'split': split, 'text': text,
            'audio_path': final.name, 'audio_sha256': _sha256_file(final),
            'relevant': relevant,
        }

    # -- manifest -----------------------------------------------------------

    def write_manifest(self, queries: list[dict], tool_versions: dict) -> Path:
        corpus = hashlib.sha256()
        for file_id in sorted(self.files):
            corpus.update(f'{file_id}:{self.files[file_id]["sha256"]}'.encode('utf-8'))
        requests_path = self.output / 'document_requests.jsonl'
        with requests_path.open('w', encoding='utf-8') as handle:
            for item in self.document_requests:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        manifest = {
            'schema_version': 1,
            'generator_version': GENERATOR_VERSION,
            'fixture_notes': FIXTURE_NOTES,
            'voices': {'queries': QUERY_VOICE, 'narrations': NARRATION_VOICE},
            'tool_versions': tool_versions,
            'corpus_checksum': corpus.hexdigest(),
            'files': [self.files[key] for key in sorted(self.files)],
            'queries': queries,
            'document_requests_path': requests_path.name,
        }
        path = self.output / 'manifest.json'
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        return path


def _view_dict(view: MemoryView) -> dict:
    from dataclasses import asdict

    return asdict(view)


def _request_dict(request: EmbedRequest) -> dict:
    from dataclasses import asdict

    return asdict(request)


def _tool_versions() -> dict:
    versions = {'python': sys.version.split()[0], 'font': FONT_CANDIDATES[0]}
    import pymupdf as fitz
    import PIL

    versions['pymupdf'] = getattr(fitz, '__version__', getattr(fitz, 'VersionBind', 'unknown'))
    versions['pillow'] = PIL.__version__
    ffmpeg = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=30)
    versions['ffmpeg'] = ffmpeg.stdout.decode().split()[2]
    espeak = subprocess.run(['espeak-ng', '--version'], capture_output=True, timeout=30)
    espeak_parts = espeak.stdout.decode().replace(':', ' ').split()
    versions['espeak-ng'] = next((part for part in espeak_parts if part[0].isdigit()), 'unknown')
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Genera il corpus di valutazione')
    parser.add_argument('--output', default='tests/fixtures/ambient')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        print(f'ERROR: la directory di output non è vuota: {output}', file=sys.stderr)
        return 1
    builder = FixtureBuilder(output)
    import tempfile

    with tempfile.TemporaryDirectory(prefix='am-fixtures-') as tmp:
        workdir = Path(tmp)
        builder.build_markdown()
        builder.build_pdfs()
        builder.build_images()
        builder.build_videos(workdir)
        queries = builder.build_queries(workdir)
        manifest = builder.write_manifest(queries, _tool_versions())
    print(json.dumps({'manifest': str(manifest), 'files': len(builder.files), 'queries': len(queries)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
