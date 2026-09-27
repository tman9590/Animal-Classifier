#!/usr/bin/env python3
"""Verify local training files after a private machine-to-machine transfer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    records = json.loads((args.root / 'work/transfer/files.json').read_text())
    failed = []
    for record in records:
        path = args.root / record['path']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            failed.append(record['path'])
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != record['sha256']:
            failed.append(record['path'])
    if failed:
        raise SystemExit(f'Transfer verification failed for {len(failed)} files: {failed[:10]}')
    print(f'Verified all {len(records)} checkpoint files')


if __name__ == '__main__':
    main()
