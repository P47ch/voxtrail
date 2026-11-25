"""File publication and exclusive access for a transcription job."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import tempfile


JOB_STATE_NAME = "job_state.json"
EXPORT_NAMES = (
    "speaker_reconciliation.json",
    "words_with_speakers.json",
    "transcript.json",
    "transcript.txt",
    "transcript.srt",
    "transcript.vtt",
    "quality_report.json",
    "quality_report.txt",
)


class JobBusyError(RuntimeError):
    pass


@contextmanager
def job_lock(output_dir: Path):
    with (output_dir / ".voxtrail.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise JobBusyError(f"Job directory is already in use: {output_dir}") from exc
        # Closing the descriptor releases the lock even if the process exits.
        yield


@contextmanager
def atomic_text_file(path: Path, *, newline: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            encoding="utf-8", newline=newline, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_exports(staged_dir: Path, output_dir: Path) -> None:
    for name in EXPORT_NAMES:
        if not (staged_dir / name).is_file():
            raise FileNotFoundError(f"Missing staged export: {name}")

    # Remove the old set before publishing the new one. If publishing stops
    # partway through, the visible files still belong to a single generation.
    for name in EXPORT_NAMES:
        (output_dir / name).unlink(missing_ok=True)
    for name in EXPORT_NAMES:
        (staged_dir / name).replace(output_dir / name)
