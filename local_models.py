"""Offline checks shared by model setup and transcription."""

import hashlib
from pathlib import Path


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
