"""Adapter ASR locale opzionale e preparazione del modello (piano HUD H3).

faster_whisper si importa SOLO nei rami local/prepare: la modalità off e
tutta la suite senza extra asr funzionano senza il pacchetto installato.
Nessuna rete nel preflight; il download avviene solo nel comando prepare.
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ASR_REPO = "Systran/faster-whisper-base"
ASR_MODEL_REPOS = {"base": ASR_REPO}
MODEL_MANIFEST_NAME = "manifest.json"
MODEL_WEIGHTS_NAME = "model.bin"


@dataclass(frozen=True)
class AsrParams:
    """Parametri operativi deterministici, registrati nel manifest export."""

    beam_size: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False


class LocalTranscriber:
    """Adapter faster-whisper CPU int8: un'istanza per esportazione.

    Il modello si carica al primo uso; i parametri operativi sono fissi di
    classe (AsrParams) e la lingua arriva dalla configurazione export.
    """

    def __init__(self, model_dir: str, language: str):
        self.model_dir = model_dir
        self.language = language
        self._model = None

    def transcribe(self, wav_path: str) -> str:
        """Trascrive un file audio; testo del modello ASR, non citazioni garantite."""
        if self._model is None:
            from faster_whisper import WhisperModel  # import lazy: solo ramo local

            self._model = WhisperModel(self.model_dir, device="cpu", compute_type="int8")
        params = AsrParams()
        segments, _info = self._model.transcribe(
            wav_path,
            language=self.language,
            beam_size=params.beam_size,
            temperature=params.temperature,
            condition_on_previous_text=params.condition_on_previous_text,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def read_model_manifest(model_dir: str) -> dict | None:
    """Legge il manifest del modello preparato; solo filesystem, senza rete."""
    path = Path(model_dir) / MODEL_MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def preflight_local(model_dir: str | None) -> tuple[bool, str]:
    """Verifica directory, manifest e pesi del modello locale, senza rete."""
    if not model_dir:
        return False, "directory modello non configurata"
    directory = Path(model_dir)
    if not directory.is_dir():
        return False, f"directory modello mancante: {model_dir}"
    manifest = read_model_manifest(model_dir)
    if manifest is None:
        return False, f"manifest modello mancante ({MODEL_MANIFEST_NAME}): riesegui prepare"
    if not (directory / MODEL_WEIGHTS_NAME).is_file():
        return False, f"pesi modello mancanti ({MODEL_WEIGHTS_NAME}): riesegui prepare"
    revision = manifest.get("revision", "")
    if not revision:
        return False, "revisione non registrata nel manifest: riesegui prepare"
    return True, f"{manifest.get('repo', ASR_REPO)}@{revision}"


def _download_snapshot(output_dir: Path, model: str) -> str:
    """Scarica la revisione risolta (commit fisso) del repo ASR; ritorna il commit."""
    from huggingface_hub import HfApi, snapshot_download  # import lazy: solo preparazione

    repo = ASR_MODEL_REPOS[model]
    revision = HfApi().model_info(repo).sha
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(output_dir))
    return revision


def prepare_model(model: str, output_dir: str) -> dict:
    """Prepara esplicitamente il modello: mai avviato dal click export."""
    if model not in ASR_MODEL_REPOS:
        raise ValueError(
            f"ASR_MODEL_INVALID: modello non supportato: {model!r}; disponibili: {sorted(ASR_MODEL_REPOS)}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    revision = _download_snapshot(output, model)
    try:
        import faster_whisper

        asr_version = faster_whisper.__version__
    except ImportError:
        asr_version = "assente"
    manifest = {
        "schema_version": 1,
        "model": model,
        "repo": ASR_MODEL_REPOS[model],
        "revision": revision,
        "downloaded_at": time.time(),
        "versions": {"faster_whisper": asr_version, "python": platform.python_version()},
    }
    (output / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.hud.transcriber", description="Preparazione modello ASR locale")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="scarica il modello alla revisione risolta (richiede extra asr)")
    prepare.add_argument("--model", default="base", help="modello faster-whisper (default: base)")
    prepare.add_argument("--output", required=True, help="directory di destinazione, es. ./data/models/asr")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_model(args.model, args.output)
        print(f"modello {manifest['model']} @ {manifest['revision']} pronto in {args.output}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
