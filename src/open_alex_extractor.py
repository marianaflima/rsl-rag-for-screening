import os
import json
import pyalex
from dataclasses import asdict
from datetime import date, datetime
from dotenv import load_dotenv
from itertools import chain
from pathlib import Path
from pyalex import Works, OpenAlexResponseList
from models import AcademicRecord

RAW_PATH = "data/raw/records-openalex.json"
TRANSFORMATION_PATH = "data/transformation/records-openalex.json"

def setup_open_alex() -> None:
    load_dotenv()

    pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")


def search_open_alex(search_string: str, first_publication_date: str, last_publication_date: str | None = None) -> OpenAlexResponseList:
    current_year = date.today().year
    first_pub_year = date.fromisoformat(first_publication_date).year
    last_pub_year = date.fromisoformat(last_publication_date).year or None

    query = (
        Works()
        .search(f"{search_string}")
        .filter(is_oa=True, first_publication_year=f"{first_pub_year}", last_publication_year=f"{last_pub_year or current_year}")
    )

    response = [record for record in chain(*query.paginate(per_page=100))]

    file_path = Path(RAW_PATH)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(response, file, default=json_default, indent=4, ensure_ascii=False)


def json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return asdict(obj)


def get_authors(data) -> str | None:
    raw_authors_list = data.get("authorships", [])

    authors_list = [a.get("raw_author_name", "") for a in raw_authors_list]

    return authors_list


def get_abstract_text(data) -> str | None:
    inv_index = data["abstract_inverted_index"]

    if inv_index is not None:
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))
    else:
        return "abstract index not available, try accessing the article if necessary"


def get_main_source(data) -> str | None:
    return data.get("primary_location").get("raw_source_name")


def get_keywords(data) -> list[str] | None:
    raw_keywords = data.get("keywords", {})

    keywords = [k.get("display_name", "") for k in raw_keywords]

    return keywords


def get_topics(data) -> list[str] | None:
    raw_topics = data.get("mesh", [])
    topics = []

    if None in raw_topics:
        raw_topics = data.get("concepts", [])
        topics = [c.get("display_name", "") for c in raw_topics]
    else:
        topics = [
            m.get("descriptor_name", "")
            for m in raw_topics
            if m.get("is_major_topic") is True
        ]
        topics = list(set(topics))

    return topics


def format_output() -> None:
    print(RAW_PATH)

    with open(RAW_PATH) as f:
        result_list = json.load(f)
    records = []

    for result in result_list:
        record = {
            "source": get_main_source(result),
            "doi": result.get("doi", ""),
            "title": result.get("title", ""),
            "abstract": get_abstract_text(result),
            "keywords": get_keywords(result),
            "topics": get_topics(result),
            "authors": get_authors(result),
            "pub_date": result.get("publication_date", ""),
            "pdf_url": result.get("primary_location", {}).get("pdf_url", ""),
            "article_type": result.get("type", ""),
            "provider": "openalex"
        }

        records.append(AcademicRecord(**record))

    file_path = Path(TRANSFORMATION_PATH)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(records, file, default=json_default, indent=4, ensure_ascii=False)

def extract_from_openalex(query: str, first_pub_date: str, last_pub_date: str | None = None):
    setup_open_alex()
    extract_from_openalex(query, first_pub_date, last_pub_date or None)
    format_output()