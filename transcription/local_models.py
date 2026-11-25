"""Offline checks shared by model setup and transcription."""

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
WHISPER_MODEL_PATH = MODELS_DIR / "whisper" / "large-v3-turbo.pt"
PYANNOTE_MODEL_PATH = MODELS_DIR / "pyannote" / "speaker-diarization-community-1"


# This is the Turbo checkpoint published by the pinned Whisper release.
WHISPER_SHA256 = "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a"
PYANNOTE_REQUIRED_ASSETS = (
    "config.yaml",
    "embedding/pytorch_model.bin",
    "segmentation/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def whisper_model_error(path: Path) -> str | None:
    try:
        if file_sha256(path) != WHISPER_SHA256:
            return "checksum mismatch"
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    return None


def missing_pyannote_assets(directory: Path) -> list[str]:
    missing = []
    for asset in PYANNOTE_REQUIRED_ASSETS:
        try:
            with (directory / asset).open("rb") as handle:
                prefix = handle.read(1024)
        except OSError:
            prefix = b""
        if not prefix or prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            missing.append(asset)
    return missing


def validate_local_models(*, require_pyannote: bool = True) -> None:
    errors = []
    setup_flag = "" if require_pyannote else " --whisper-only"
    setup = f"`bash setup_models.sh{setup_flag}` or `setup_models.cmd{setup_flag}` on Windows"

    whisper_error = whisper_model_error(WHISPER_MODEL_PATH)
    if whisper_error:
        errors.append(f"Whisper Turbo model {whisper_error}: {WHISPER_MODEL_PATH}")

    if require_pyannote:
        missing = missing_pyannote_assets(PYANNOTE_MODEL_PATH)
        if missing:
            assets = ", ".join(missing)
            errors.append(f"pyannote Community-1 model missing/incomplete: {PYANNOTE_MODEL_PATH} ({assets})")

    if errors:
        errors.append(f"Run {setup} to install or repair the local models.")
        raise RuntimeError("\n\n".join(errors))
