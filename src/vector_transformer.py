from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from models import AcademicRecord
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import json

COLLECTION_NAME = "classic-rag-rsl"
VECTOR_SIZE = 768

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def json_to_markdown_doc(record: AcademicRecord) -> list[Document]:
    conteudo_md = f"""# {record.get('title', 'No title provided')}

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
        "autores": ", ".join(record.get("authors", [])),
        "pub_date": record.get("pub_date"),
        "url": record.get("pdf_url"),
        "provider": record.get("provider")
    }

    return Document(page_content=conteudo_md, metadata=metadata)


def setup_qdrant() -> None:

    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' já existe.")
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(
            f"Collection '{COLLECTION_NAME}' criada com sucesso (Dimensão: {VECTOR_SIZE}, Métrica: Cosine)."
        )


def chunk_documents(documents: list[Document]) -> list[Document]:
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    chunked_docs = []

    for doc in documents:
        # Fatia o markdown do documento
        chunks = markdown_splitter.split_text(doc.page_content)
        
        # Preserva os metadados originais do artigo em cada chunk
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
            chunked_docs.append(chunk)

    print(f"Total chunks generated: {len(chunked_docs)}")
    print("Chunk example:\n", chunked_docs[0].page_content)
    print("Chunk example metadata:\n", chunked_docs[0].metadata)

    return chunked_docs

def assembly_retriever(chunks: list[Document]) -> EnsembleRetriever:

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:v1.5", base_url="http://localhost:7869"
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        url="http://localhost:6333",
    )
    vector_store = QdrantVectorStore.from_existing_collection(COLLECTION_NAME)

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[
            0.4,
            0.6,
        ],
    )

    return ensemble_retriever


def assembly_rag_chain(retriever: EnsembleRetriever):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=None,
        timeout=500,
        max_retries=2,
    )

    template = """
    You are an academic records expert. Use ONLY the data provided to answer.
    If the answer isn't in the context, say "There's is no information about this in the provided data"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def initialize_rory(rag_chain: RunnableSerializable):

    print("=" * 60)
    print(" " * 5, "RORY: Research Orchestration and Review sYstem", " " * 5)
    print("=" * 60)

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nRORY: {response}")

def assembly_vector_rag():
    setup_qdrant()

    with open("data/final/all-records.json") as f:
        json_documents = json.load(f)

    documents = [json_to_markdown_doc(r) for r in json_documents]
    documents = documents[0:10]

    chunks = chunk_documents(documents)
    retriever = assembly_retriever(chunks)

    rag_chain = assembly_rag_chain(retriever)

    initialize_rory(rag_chain)




