from datetime import date, datetime
from dataclasses import asdict
from random import randint

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

# from src.models import AcademicRecord
from models import AcademicRecord
from langchain_core.documents import Document


def json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return asdict(obj)


def json_to_markdown_doc(record: AcademicRecord) -> list[Document]:
    conteudo_md = f"""# {record.get('title', 'No title provided')}
## Authors
{", ".join(record.get("authors"))}
    
## Abstract
{record.get('abstract', 'No abstract provided')}

## Keywords
{record.get('keywords', 'No keywords provided')}

## Topics
{record.get('topics', 'No topics provided')}
"""
    metadata = {
        "doi": record.get("doi"),
        "titulo": record.get("title"),
        "pub_date": record.get("pub_date"),
        "url": record.get("pdf_url"),
        "provider": record.get("provider"),
    }

    return Document(page_content=conteudo_md, metadata=metadata)


def mount_random_article_test_list(data: list[dict]) -> list[dict]:
    test_docs = []
    for i in range(10):
        rand = randint(0, len(data) - 1)
        doc = data[rand]
        test_docs.append(doc)

    return test_docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents=documents)

    print("=" * 20)
    print(f"Total chunks generated: {len(chunks)}")
    print("=" * 20)

    return chunks
