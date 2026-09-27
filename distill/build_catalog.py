#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import Species, fetch_species, write_catalog


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a regional iNaturalist species vocabulary")
    parser.add_argument("--config", type=Path, default=ROOT / "distill" / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "north-carolina-species.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    species = fetch_species(
        config["inaturalist_place_id"],
        config["iconic_taxa"],
        config["maximum_species"],
        config["minimum_inaturalist_observations"],
    )
    by_taxon_id = {item.taxon_id: item for item in species}
    for required in config.get("required_species", []):
        taxon_id = int(required["taxon_id"])
        observation_count = by_taxon_id[taxon_id].observation_count if taxon_id in by_taxon_id else 0
        by_taxon_id[taxon_id] = Species(
            **required,
            observation_count=observation_count,
        )
    species = sorted(by_taxon_id.values(), key=lambda item: (-item.observation_count, item.scientific_name))
    write_catalog(args.output, config["region"], config["inaturalist_place_id"], species)
    print(f"Wrote {len(species)} species plus unknown to {args.output}")


if __name__ == "__main__":
    main()
