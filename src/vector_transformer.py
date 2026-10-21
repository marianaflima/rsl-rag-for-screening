from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

COLLECTION_NAME = "classic-rag-rsl"
VECTOR_SIZE = 768

def convert_embeddings():

    client = QdrantClient(url="http://localhost:6333")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:v1.5",
        base_url="http://localhost:7869"
    )

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

    script_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        add_start_index=True,
        separators=["""["(""", """)"]""", "", """)", """, "\n\n", "\n", " ", ""],
    )

    all_chunks = []

    with open("data/final/all-records.md") as f:
        md = f.read()
        document = Document(page_content=md)

    chunks = script_splitter.split_documents([document])
    all_chunks.extend(chunks)
    print(f"Loaded content and split into {len(all_chunks)} chunks")

    vector_store = QdrantVectorStore.from_documents(
        documents=all_chunks,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        url="http://localhost:6333"
    )

def main():
    llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=None,
            timeout=500,
            max_retries=2,
        )

    vector_store = QdrantVectorStore.from_existing_collection(COLLECTION_NAME)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    template = """
    You are an academic revisioner expert. Use ONLY the data provided to answer.
    If the answer isn't in the context, say "There's is no information about this in the provided data"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("=" * 60)
    print(" " * 15, "RORY: Research Orchestration and Review sYstem", " " * 15)
    print("=" * 60)

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nRORY: {response}")


main()
