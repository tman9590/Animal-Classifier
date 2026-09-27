from __future__ import annotations

import json
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


API = "https://api.inaturalist.org/v1"
USER_AGENT = "Scrypted-Animal-Classifier/0.2 (+https://github.com/tman9590/scrypted-animal-classifier)"


@dataclass(frozen=True)
class Species:
    taxon_id: int
    scientific_name: str
    common_name: str
    iconic_taxon: str
    observation_count: int

    @property
    def label(self) -> str:
        return self.common_name

    @property
    def prompts(self) -> tuple[str, str]:
        return (
            f"a camera trap photo of {self.common_name}, species {self.scientific_name}",
            f"a photograph of the animal species {self.scientific_name}",
        )


def api_get(path: str, params: dict, opener: Callable = urllib.request.urlopen, retries: int = 6) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params, doseq=True)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with opener(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == retries - 1:
                raise
            retry_after = float(error.headers.get("Retry-After") or 2 ** attempt)
            time.sleep(min(60, retry_after))
    raise RuntimeError("unreachable")


def fetch_species(
    place_id: int,
    iconic_taxa: list[str],
    limit: int,
    minimum_observations: int = 20,
    opener: Callable = urllib.request.urlopen,
) -> list[Species]:
    candidates: list[Species] = []
    for iconic_taxon in iconic_taxa:
        page = 1
        while len(candidates) < limit:
            payload = api_get(
                "observations/species_counts",
                {
                    "place_id": place_id,
                    "taxon_id": 1,
                    "quality_grade": "research",
                    "rank": "species",
                    "iconic_taxa[]": iconic_taxon,
                    "per_page": 200,
                    "page": page,
                    "locale": "en",
                },
                opener,
            )
            results = payload.get("results", [])
            if not results:
                break
            for result in results:
                taxon = result.get("taxon") or {}
                observation_count = int(result.get("count") or 0)
                if observation_count < minimum_observations:
                    continue
                if taxon.get("iconic_taxon_name") != iconic_taxon or taxon.get("rank") != "species":
                    continue
                candidates.append(
                    Species(
                        taxon_id=int(taxon["id"]),
                        scientific_name=taxon["name"],
                        common_name=taxon.get("preferred_common_name") or taxon["name"],
                        iconic_taxon=iconic_taxon,
                        observation_count=observation_count,
                    )
                )
                if len(candidates) >= limit:
                    break
            page += 1
            if page > (int(payload.get("total_results", 0)) + 199) // 200:
                break
    return sorted(candidates, key=lambda item: (-item.observation_count, item.scientific_name))[:limit]


def write_catalog(path: Path, region: str, place_id: int, species: list[Species]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "region": region,
        "inaturalist_place_id": place_id,
        "species": [asdict(item) for item in species],
        "labels": [item.label for item in species] + ["unknown"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_catalog(path: Path) -> tuple[dict, list[Species]]:
    payload = json.loads(path.read_text())
    return payload, [Species(**item) for item in payload["species"]]
