"""Mapping multimodale e backend di embedding (Task I2).

`build_model_input` traduce un EmbedRequest nell'input canonico del modello
(prefissi di attività, placeholder <|image|>/<|audio|> in ordine), verificabile
senza pesi. `EmbeddingGemmaBackend` carica esattamente model_id/revision dello
SpaceSpec, CPU float32 o CUDA bf16 (mai float16), conta il budget totale con
il tokenizer e proietta l'output su 256 dimensioni unitarie.
"""

import hashlib
import os
import re

import numpy as np

from src.contracts import EMBED_DIM, EmbedRequest, SpaceSpec, validate_vector
from src.inference.client import EmbeddingError

# Limiti del contratto e del modello (README google/embeddinggemma-2 §Multimodal).
QUERY_PREFIX = "task: search result | query: "
MAX_TEXT_TOKENS = 6000
MAX_AUDIO_SEC = 45.0
AUDIO_SAMPLERATE = 16000
TOKENS_PER_IMAGE = 280
TOKENS_PER_AUDIO_SEC = 25
MODEL_TOTAL_BUDGET = 8192
MIN_NORM = 1e-12


def build_model_input(req: EmbedRequest) -> str | dict:
    """Input canonico per model.encode secondo le regole del piano (I2).

    Query testo: prefisso di ricerca. Query audio pura: dict senza prefisso
    testuale. Documento: `title: {header o none} | text: {corpo}` seguito dai
    placeholder in ordine (immagini nell'ordine ricevuto, poi audio).
    """
    if req.is_query:
        if req.image_paths:
            raise EmbeddingError("INVALID_REQUEST", "query con immagini non supportata: un canale solo")
        if req.audio_path is not None:
            if req.text_payload:
                raise EmbeddingError("INVALID_REQUEST", "query con testo e audio insieme")
            return {"audio": req.audio_path}
        return QUERY_PREFIX + req.text_payload
    parts = [f"title: {req.context_header or 'none'} | text: {req.text_payload}"]
    parts.extend(["<|image|>"] * len(req.image_paths))
    if req.audio_path is not None:
        parts.append("<|audio|>")
    doc_text = " ".join(parts)
    if not req.image_paths and req.audio_path is None:
        return doc_text
    model_input: dict = {"text": doc_text}
    if req.image_paths:
        model_input["image"] = list(req.image_paths)
    if req.audio_path is not None:
        model_input["audio"] = req.audio_path
    return model_input


def project_output(raw: np.ndarray) -> list[float]:
    """Proietta l'output del modello su EMBED_DIM unitarie; nessun flatten arbitrario."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 2 and raw.shape[0] == 1:
        raw = raw[0]  # elimina solo la dimensione batch esplicita
    if raw.shape != (EMBED_DIM,) or not np.isfinite(raw).all():
        raise ValueError("INVALID_MODEL_OUTPUT")
    norm = float(np.linalg.norm(raw))
    if norm < MIN_NORM:
        raise ValueError("INVALID_VECTOR_NORM")
    result = (raw / norm).tolist()
    validate_vector(result)
    return result


def budget_check(text_tokens: int, n_images: int, audio_sec: float) -> None:
    """Budget totale condiviso: testo + immagini + audio nel contesto 8192."""
    total = text_tokens + TOKENS_PER_IMAGE * n_images + TOKENS_PER_AUDIO_SEC * audio_sec
    if text_tokens > MAX_TEXT_TOKENS or total > MODEL_TOTAL_BUDGET:
        raise EmbeddingError(
            "INPUT_TOO_LARGE",
            f"budget modello superato: testo {text_tokens}/{MAX_TEXT_TOKENS} token, totale {total:.0f}/{MODEL_TOTAL_BUDGET}",
        )


def _detect_modalities(model) -> set[str]:
    """Modalità effettivamente caricate nel modello, prima di accettare richieste."""
    modalities = {"text"}
    processor = getattr(model, "processor", None)
    if processor is not None:
        if (
            getattr(processor, "image_processor", None) is not None
            or getattr(processor, "video_processor", None) is not None
        ):
            modalities.add("image")
        if getattr(processor, "feature_extractor", None) is not None:
            modalities.add("audio")
    return modalities


def _resolve_dtype(device: str):
    """bf16 solo su CUDA che lo supporta, altrimenti float32. Mai float16."""
    if device == "cuda":
        import torch

        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
    return None, "float32"


class EmbeddingGemmaBackend:
    """Backend reale: caricamento alla revisione dello SpaceSpec, nessun fallback."""

    def __init__(self, space: SpaceSpec, device: str = "auto"):
        self._space = space
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        _, dtype_name = _resolve_dtype(device)
        self._model = self._load_model(space, device)
        self._modalities = _detect_modalities(self._model)
        self.info = {
            "model_id": space.model_id,
            "revision": space.revision,
            "device": self._device,
            "dtype": dtype_name,
            "modalities": sorted(self._modalities),
        }
        for module_name in ("torch", "transformers", "sentence_transformers"):
            try:
                module = __import__(module_name)
                self.info[module_name] = getattr(module, "__version__", "sconosciuto")
            except ImportError:
                self.info[module_name] = "non disponibile"

    @classmethod
    def _load_model(cls, space: SpaceSpec, device: str):
        """Punto di sostituzione per i test: carica SentenceTransformer ai pesi reali."""
        from sentence_transformers import SentenceTransformer

        dtype, _ = _resolve_dtype(device)
        model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        return SentenceTransformer(
            space.model_id,
            revision=space.revision,
            device=device,
            model_kwargs=model_kwargs,
        )

    def _token_count(self, text: str) -> int:
        if not text:
            return 0
        encoded = self._model.tokenizer(text)
        return len(encoded["input_ids"])

    def _check_media(self, req: EmbedRequest) -> float:
        """Esistenza, formato e limiti dei media; ritorna i secondi audio validi."""
        if req.image_paths and "image" not in self._modalities:
            raise EmbeddingError("MODEL_UNAVAILABLE", "modello senza encoder immagini")
        for path in req.image_paths:
            if not os.path.isfile(path):
                raise EmbeddingError("INVALID_MEDIA", f"immagine inesistente: {path}")
        if req.audio_path is None:
            return 0.0
        if "audio" not in self._modalities:
            raise EmbeddingError("MODEL_UNAVAILABLE", "modello senza encoder audio")
        if not os.path.isfile(req.audio_path):
            raise EmbeddingError("INVALID_MEDIA", f"audio inesistente: {req.audio_path}")
        import soundfile

        try:
            info = soundfile.info(req.audio_path)
        except Exception as exc:
            raise EmbeddingError("INVALID_MEDIA", f"audio illeggibile: {req.audio_path}: {exc}") from exc
        if info.samplerate != AUDIO_SAMPLERATE:
            raise EmbeddingError("INVALID_MEDIA", f"audio non {AUDIO_SAMPLERATE} Hz: {info.samplerate}")
        if info.channels != 1:
            raise EmbeddingError("INVALID_MEDIA", f"audio non mono: {info.channels} canali")
        if info.duration > MAX_AUDIO_SEC:
            raise EmbeddingError("INPUT_TOO_LARGE", f"audio {info.duration:.1f}s oltre {MAX_AUDIO_SEC}s")
        return info.duration

    def embed(self, req: EmbedRequest) -> list[float]:
        model_input = build_model_input(req)
        audio_sec = self._check_media(req)
        text = model_input.get("text", "") if isinstance(model_input, dict) else model_input
        budget_check(self._token_count(text), len(req.image_paths), audio_sec)
        try:
            import torch

            with torch.inference_mode():
                raw = self._model.encode(model_input, truncate_dim=EMBED_DIM, normalize_embeddings=True)
            return project_output(raw)
        except EmbeddingError:
            raise
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            raise EmbeddingError("MODEL_UNAVAILABLE", f"{type(exc).__name__}: {exc}") from exc


# ------------------------------------------------------------------ Task I3: mock

MOCK_MODEL_ID = "semantic-hash-mock"
MOCK_REVISION = "1"
_SENTINEL = "empty:payload"
_TOKEN_PATTERN = r"[\wàèéìòùáűőäöüßçñ]+"
MEDIA_PREFIX = "media"


def mock_space() -> SpaceSpec:
    """SpaceSpec dedicato del mock: mai confrontabile con lo spazio reale."""
    return SpaceSpec(backend="mock", model_id=MOCK_MODEL_ID, revision=MOCK_REVISION)


class SemanticHashMock:
    """Mock dichiarato (--backend mock): feature hashing deterministico.

    Unigrammi e bigrammi del testo in minuscolo, marcatori di tipo e proprietà
    dei media (basename), token sentinella per payload senza feature. Nessuna
    pretesa di comprensione del contenuto: l'output non sostituisce la prova
    semantica e non supera la Fase 1.
    """

    def __init__(self, space: SpaceSpec | None = None):
        space = space or mock_space()
        if space.model_id != MOCK_MODEL_ID or space.revision != MOCK_REVISION or space.backend != "mock":
            raise EmbeddingError(
                "INVALID_REQUEST",
                "SemanticHashMock richiede lo SpaceSpec del mock (semantic-hash-mock/1/mock)",
            )
        self._space = space

    def features(self, req: EmbedRequest) -> list[str]:
        tokens = re.findall(_TOKEN_PATTERN, req.text_payload.lower())
        features = tokens + [f"{first}~{second}" for first, second in zip(tokens, tokens[1:])]
        for path in req.image_paths:
            features.append(f"{MEDIA_PREFIX}:image")
            features.append(f"img:{os.path.basename(path)}")
        if req.audio_path is not None:
            features.append(f"{MEDIA_PREFIX}:audio")
            features.append(f"aud:{os.path.basename(req.audio_path)}")
        if not features:
            features = [_SENTINEL]
        return features

    def _hash_vector(self, features: list[str]) -> np.ndarray:
        vector = np.zeros(EMBED_DIM, dtype=np.float64)
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % EMBED_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector

    def embed(self, req: EmbedRequest) -> list[float]:
        features = self.features(req)
        vector = self._hash_vector(features)
        # collisioni con segni opposti possono annullare il vettore (es. query
        # audio con due sole feature): salt deterministico, stesse feature ->
        # stesso vettore, senza distruggere la similarità fra input diversi
        salt = 0
        while float(np.linalg.norm(vector)) < MIN_NORM and salt < 8:
            salt += 1
            vector = self._hash_vector(features + [f"salt:{salt}"])
        norm = float(np.linalg.norm(vector))
        if norm < MIN_NORM:
            raise EmbeddingError("MODEL_UNAVAILABLE", "INVALID_VECTOR_NORM: vettore mock degenere")
        result = (vector / norm).tolist()
        validate_vector(result)
        return result

    @property
    def info(self) -> dict:
        """Stesso contratto di EmbeddingGemmaBackend.info per report uniformi."""
        return {
            "model_id": self._space.model_id,
            "revision": self._space.revision,
            "device": "cpu",
            "dtype": "float64",
            "modalities": ["audio", "image", "text"],
        }
