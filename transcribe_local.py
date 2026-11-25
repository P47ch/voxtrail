"""Run the local transcription pipeline."""

import sys

from transcription import runtime
from transcription.cli import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        runtime.clear_gpu()
        print("\nInterrupted. Completed chunk checkpoints are safe; rerun the same command to resume.")
        sys.exit(130)
