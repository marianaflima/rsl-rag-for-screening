from dataclasses import dataclass
from datetime import date

@dataclass
class AcademicRecord:
    source: str
    doi: str | None
    title: str
    abstract: str
    keywords: list[str]
    topics: list[str] | None
    authors: list[str]
    pub_date: date | None
    pdf_url: str | None
    article_type: str | None
    provider: str
