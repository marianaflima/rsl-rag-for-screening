from dataclasses import asdict
from datetime import date, datetime
from models import AcademicRecord
from pathlib import Path
import html
import json
import re
import requests

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
RAW_PATH = "data/raw/records-europepmc.json"
TRANSFORMATION_PATH = "data/transformation/records-europepmc.json"

def search_europe_pmc(
    query: str,
    first_publication_date: str,
    last_publication_date: str | None = None,
    page_size: int = 500,
) -> None:
    current_date = date.today().isoformat()

    url_query = (
        f"({query}) "
        f"HAS_FREE_FULLTEXT:Y "
        f"FIRST_PDATE:[{first_publication_date} TO {last_publication_date or current_date}]"
    )

    count_per_list = 0
    cursor_mark = "*"
    results = []

    print("\n\nComeçando obtenção dos artigos no Europe PMC...\n")
    while True:

        params = {
            "query": url_query,
            "format": "json",
            "resultType": "core",
            "synonym": "YES",
            "cursorMark": cursor_mark,
            "sort": "P_PDATE_D desc",
            "pageSize": page_size,
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=300,
        )
        response.raise_for_status()

        data = response.json()
        hit_count = data["hitCount"]
        count_per_list += len(data["resultList"]["result"])
        print(f"{count_per_list}/{hit_count} artigo(s) obtidos | {((count_per_list/hit_count)*100):.0f} %")

        resp_list = data["resultList"]["result"]

        results.extend(resp_list)

        next_cursor = data.get("nextCursorMark", "") or cursor_mark

        if cursor_mark == next_cursor:
            break

        cursor_mark = next_cursor

    print("Obtenção concluída")

    file_path = Path(RAW_PATH)
    file_path.parent.mkdir(parents=True, exist_ok=True)
        
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(results, file, default=json_default, indent=4, ensure_ascii=False)


def get_record_source(data) -> str | None:
    if data.get("journalInfo", {}).get("journal", {}).get("title", "") is not None:
        return data.get("journalInfo", {}).get("journal", {}).get("title", "")
    elif data.get("bookOrReportDetails", {}).get("publisher", "") is not None:
        return data.get("bookOrReportDetails", {}).get("publisher", "")
    else:
        return None


def get_best_url(data) -> str | None:
    priority = ["pdf", "html", "doi"]

    urls = data.get("fullTextUrlList", {}).get("fullTextUrl", [])

    for document_style in priority:
        for item in urls:
            if (
                item.get("availabilityCode") in ("OA", "F")
                and item.get("documentStyle") == document_style
            ):
                return item.get("url")

    return None

def get_article_type_string(data) -> str | None:
    input_type_list = data.get("pubTypeList", {})

    types = input_type_list.get("pubType", [])
    types = (t.lower() for t in types)

    return "|".join(types)


def get_publication_date(data) -> date | None:
    pub_date = data.get("firstPublicationDate", "")
    return datetime.strptime(pub_date, "%Y-%m-%d").date()


def get_keywords(data) -> list[str] | None:
    keyword_list = data.get("keywordList", {})
    
    if keyword_list != {}:
        return keyword_list.get("keyword")

    return None

def get_topics(data) -> list[str] | None: 
    mesh_heading_list_header = data.get("meshHeadingList", {})
    
    if mesh_heading_list_header != {}:
        mesh_headings_list = mesh_heading_list_header.get("meshHeading")
        mesh_headings = [m.get("descriptorName", "") for m in mesh_headings_list if m.get("majorTopic_YN", "") == "Y"]

        return mesh_headings

    return None

def get_authors(data) -> list[str] | None:

    raw_authors_list = data.get("authorList", {})

    if raw_authors_list != {}:
        authors = [f"{a.get("firstName")} {a.get("lastName")}" for a in raw_authors_list.get("author")]
        return authors

    return None

def json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return asdict(obj)

def strip_html(text: str | None) -> str | None:
    TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")
    if text is None:
        return None
    cleaned = TAG_PATTERN.sub("", text)      
    cleaned = html.unescape(cleaned)         
    cleaned = re.sub(r"\s+", " ", cleaned)   
    return cleaned.strip()

def format_output() -> None:
    print(RAW_PATH)

    with open(RAW_PATH) as f:
        result_list = json.load(f)
    
    records = []

    print("Processando artigos obtidos...\n")

    for result in result_list:
        abstract = result.get("abstractText")
        abstract = strip_html(abstract)
        record = {
            "source": get_record_source(result),
            "doi": result.get("doi"),
            "title": result.get("title"),
            "abstract": abstract,
            "keywords": get_keywords(result),
            "topics": get_topics(result),
            "authors": get_authors(result),
            "pub_date": get_publication_date(result),
            "pdf_url": get_best_url(result),
            "article_type": get_article_type_string(result),
            "provider": "europepmc"
        }

        records.append(AcademicRecord(**record))

    file_path = Path(TRANSFORMATION_PATH)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open("data/transformation/records-europepmc.json", "w", encoding="utf-8") as file:
        json.dump(records, file, default=json_default, indent=4, ensure_ascii=False)

def extract_from_europe_pmc(query: str, lower_limit_date: str, upper_limit_date: str | None = None):
    search_europe_pmc(query, lower_limit_date, upper_limit_date or None)
    format_output()


