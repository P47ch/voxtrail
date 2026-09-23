#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -x .venv/bin/python ]]; then
    python_bin=.venv/bin/python
elif [[ -x venv/bin/python ]]; then
    python_bin=venv/bin/python
else
    echo "Python virtual environment not found. Create it and install requirements.txt first." >&2
    echo "See the Python environment setup section in README.md." >&2
    exit 1
fi

exec "$python_bin" setup_models.py "$@"
