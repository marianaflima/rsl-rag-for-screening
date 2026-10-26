from dataclasses import asdict
from datetime import date, datetime
from models import AcademicRecord
from pathlib import Path
import json

OPENALEX_RECORDS = "data/transformation/records-openalex.json"
EUROPEPMC_RECORDS = "data/transformation/records-europepmc.json"
ALL_RECORDS = "data/final/all-records.json"

def json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return asdict(obj)


def obtain_records() -> list[AcademicRecord]:
    with open(OPENALEX_RECORDS) as f:
        openalex_records = json.load(f)

    with open(EUROPEPMC_RECORDS) as f:
        europepmc_records = json.load(f)

    unified_records = europepmc_records + openalex_records
    unified_records = [AcademicRecord(**r) for r in unified_records]

    return unified_records


def filter_missing_values(records: list[AcademicRecord]) -> list[AcademicRecord]:
    filtered_records = [
        obj for obj in records 
        if all(getattr(obj, field) is not None or [] for field in obj.__dataclass_fields__)
    ]

    return filtered_records

def doi_title_deduplication(records: list[AcademicRecord]) -> list[AcademicRecord]:

    vistos = set()
    dedup_records = []

    for r in records:
        dupe_key = (r.doi, r.title)

        if dupe_key not in vistos:
            vistos.add(dupe_key)
            dedup_records.append(r)

    return dedup_records

def apply_transformations():
    records = obtain_records()
    print(f"\nRecuperados {len(records)} artigo(s)")
    filtered_records = filter_missing_values(records)
    print(f"\nFiltrados {len(filtered_records)} artigo(s) sem valores vazios")
    dedup_records = doi_title_deduplication(filtered_records)
    print(f"\nRetornados {len(dedup_records)} artigo(s) após deduplicação")

    file_path = Path(ALL_RECORDS)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open("data/final/all-records.json", "w", encoding="utf-8") as file:
            json.dump(dedup_records, file, default=json_default, indent=4, ensure_ascii=False)

    print(f"\nArtigos convertidos e salvos em json")
