#!/bin/sh
set -eu

cd "$(dirname "$0")"
if [ ! -f .venv/bin/activate ]; then
    python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
