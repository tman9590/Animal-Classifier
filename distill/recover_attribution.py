#!/usr/bin/env python3
"""Recover current license/credit records for already-downloaded photos only."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from catalog import api_get, read_catalog
from download_inaturalist import image_url, write_manifest

ROOT = Path(__file__).parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'work/source-images')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'species/north-carolina.json')
    args = parser.parse_args()
    _, species = read_catalog(args.catalog)
    manifest = args.source / 'attribution.csv'
    with manifest.open() as stream:
        rows = {row['file']: row for row in csv.DictReader(stream)}
    missing = [p for p in args.source.glob('*/*.jpg') if p.relative_to(args.source).as_posix() not in rows]
    recovered = 0
    for offset in range(0, len(missing), 50):
        batch = missing[offset:offset + 50]
        identifiers = ','.join(sorted({p.stem.split('-')[0] for p in batch}))
        payload = api_get('observations', {'id': identifiers, 'per_page': 200})
        observations = {str(o['id']): o for o in payload.get('results', [])}
        for path in batch:
            observation_id, photo_id = path.stem.split('-')
            observation = observations.get(observation_id)
            if not observation:
                continue
            class_index = int(path.parent.name.split('-')[0])
            item = species[class_index]
            taxon = observation.get('taxon') or {}
            ancestry = set(taxon.get('ancestor_ids') or []) | {taxon.get('id')}
            if item.taxon_id not in ancestry:
                continue  # Identification changed: do not retain a stale class assignment.
            for photo in observation.get('photos') or []:
                license_code = (photo.get('license_code') or '').lower()
                if str(photo['id']) != photo_id or license_code not in {'cc0', 'cc-by'}:
                    continue
                key = path.relative_to(args.source).as_posix()
                rows[key] = dict(file=key, class_index=class_index, taxon_id=item.taxon_id,
                                 scientific_name=item.scientific_name, common_name=item.common_name,
                                 observation_url=f'https://www.inaturalist.org/observations/{observation_id}',
                                 photo_url=image_url(photo), license=license_code,
                                 attribution=photo.get('attribution') or '')
                recovered += 1
        write_manifest(args.source, list(rows.values()))
        print(f'Recovered {recovered}; checked {min(offset + 50, len(missing))}/{len(missing)} existing photos', flush=True)
    print(f'{len(missing) - recovered} existing files remain excluded pending verified license/taxon metadata')


if __name__ == '__main__':
    main()
