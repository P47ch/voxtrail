"""Interactive, one-time download of the models used by transcribe_local.py."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import sysconfig
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WHISPER_PATH = BASE_DIR / "models" / "whisper" / "large-v3-turbo.pt"
PYANNOTE_PATH = BASE_DIR / "models" / "pyannote" / "speaker-diarization-community-1"
PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"
PYANNOTE_URL = f"https://huggingface.co/{PYANNOTE_REPO}"
SETUP_COMMAND = "bash setup_models.sh (Linux/macOS/WSL) or setup_models.cmd (Windows)"


class SetupError(Exception):
    pass


def is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def pyannote_ready() -> bool:
    config = PYANNOTE_PATH / "config.yaml"
    if not config.is_file() or is_lfs_pointer(config):
        return False
    return not any(
        is_lfs_pointer(path)
        for path in PYANNOTE_PATH.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def show_linux_build_tools() -> None:
    if not sys.platform.startswith("linux"):
        return
    compiler = os.environ.get("CC", "cc")
    python_header = Path(sysconfig.get_path("include")) / "Python.h"
    missing = []
    if not shutil.which(compiler):
        missing.append("C compiler")
    if not python_header.is_file():
        missing.append("Python development headers")
    if missing:
        print(f"Optional Triton word-timestamp acceleration: missing {', '.join(missing)} (see README.md)")
    else:
        print("Triton build tools: C compiler and Python headers present")


def show_status(*, whisper_only: bool = False) -> bool:
    whisper_ready = WHISPER_PATH.is_file() and WHISPER_PATH.stat().st_size > 1_000_000
    print(f"Whisper Turbo: {'ready' if whisper_ready else 'missing'} ({WHISPER_PATH})")
    if whisper_only:
        print("Community-1: not required for --single-speaker")
        show_linux_build_tools()
        return whisper_ready
    diarization_ready = pyannote_ready()
    print(f"Community-1: {'ready' if diarization_ready else 'missing'} ({PYANNOTE_PATH})")
    try:
        from huggingface_hub import get_token
    except ImportError:
        print("Hugging Face client: missing (install requirements.txt)")
    else:
        print(f"Hugging Face sign-in: {'available' if get_token() else 'missing'}")
    show_linux_build_tools()
    return whisper_ready and diarization_ready


def require_login(api, login, get_token, *, force: bool) -> None:
    from huggingface_hub.utils import HfHubHTTPError

    if force and os.environ.get("HF_TOKEN"):
        raise SetupError("HF_TOKEN overrides saved sign-in. Unset HF_TOKEN before using --login.")
    if force or not get_token():
        if not sys.stdin.isatty():
            raise SetupError(f"Hugging Face sign-in needs an interactive terminal. Run {SETUP_COMMAND}.")
        print("Hugging Face sign-in will offer browser login or a hidden token prompt.")
        login(skip_if_logged_in=not force)

    try:
        api.whoami(token=True)
    except HfHubHTTPError as exc:
        if exc.response is None or exc.response.status_code not in (401, 403):
            raise
        if os.environ.get("HF_TOKEN"):
            raise SetupError("The HF_TOKEN environment variable is invalid. Update or unset it before setup.") from exc
        if not sys.stdin.isatty():
            raise SetupError(f"The saved Hugging Face sign-in is invalid. Run {SETUP_COMMAND} --login.") from exc
        print("The saved Hugging Face sign-in is invalid. Please sign in again.")
        login(skip_if_logged_in=False)
        api.whoami(token=True)


def download_models(*, force_login: bool = False, whisper_only: bool = False) -> None:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise SetupError("Model setup needs internet access. Unset HF_HUB_OFFLINE and run setup again.")

    if not whisper_only:
        try:
            from huggingface_hub import HfApi, get_token, login, snapshot_download
            from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        except ImportError as exc:
            raise SetupError("Python dependencies are missing. Install requirements.txt in the project virtual environment first.") from exc

        api = HfApi()
        require_login(api, login, get_token, force=force_login)
        try:
            api.auth_check(PYANNOTE_REPO, token=True)
        except GatedRepoError as exc:
            raise SetupError(
                f"Community-1 access has not been granted to this Hugging Face account. "
                f"Open {PYANNOTE_URL}, accept its conditions, then run setup again."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise SetupError(
                f"This Hugging Face account cannot access Community-1. Check that the same account "
                f"accepted the conditions at {PYANNOTE_URL}, then run setup again."
            ) from exc

        print("Hugging Face access to Community-1 confirmed.")
    try:
        import whisper
    except ImportError as exc:
        raise SetupError("Whisper is missing. Install requirements.txt in the project virtual environment first.") from exc
    print("Checking/downloading Whisper Turbo (about 1.5 GiB)...", flush=True)
    # openai-whisper is pinned in requirements.txt; its downloader verifies the
    # checkpoint SHA256 and reuses an existing valid file.
    turbo_url = whisper._MODELS["turbo"]
    if turbo_url.rsplit("/", 1)[-1] != WHISPER_PATH.name:
        raise SetupError("The installed Whisper Turbo filename differs from the project's expected checkpoint.")
    WHISPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    whisper._download(turbo_url, str(WHISPER_PATH.parent), False)

    if whisper_only:
        print("Whisper Turbo is ready. Use --single-speaker to transcribe without pyannote or Hugging Face sign-in.")
        show_linux_build_tools()
        return

    print("Checking/downloading pyannote Community-1...", flush=True)
    snapshot_download(repo_id=PYANNOTE_REPO, local_dir=PYANNOTE_PATH, token=True)
    if not pyannote_ready():
        raise SetupError("Community-1 download is incomplete. Run setup again to resume it.")
    print("Both local models are ready. Normal transcription runs offline and needs no token.")
    show_linux_build_tools()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Whisper Turbo and, unless --whisper-only is set, Community-1.")
    parser.add_argument("--check", action="store_true", help="Show local setup status without downloading")
    parser.add_argument("--login", action="store_true", help="Sign in again before checking access")
    parser.add_argument("--whisper-only", action="store_true", help="Set up only Whisper Turbo for --single-speaker; no Hugging Face sign-in")
    args = parser.parse_args()
    if args.check and args.login:
        parser.error("--check and --login cannot be used together")
    if args.login and args.whisper_only:
        parser.error("--login is unnecessary with --whisper-only")
    if args.check:
        return 0 if show_status(whisper_only=args.whisper_only) else 1
    try:
        download_models(force_login=args.login, whisper_only=args.whisper_only)
    except KeyboardInterrupt:
        print("\nSetup canceled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
