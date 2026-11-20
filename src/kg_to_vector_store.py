from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import numpy as np
import os
import uuid

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from utils import format_docs

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
QDRANT_URI = os.environ["QDRANT_URI"]
NON_WEIGHTED_KG_COLLECTION_NAME = "kg-vector-rag-rsl"
WEIGHTED_KG_COLLECTION_NAME = "weighted-kg-vector-rag-rsl"
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
VECTOR_SIZE = 768

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:v1.5",
    base_url="http://localhost:7869",
    dimensions=VECTOR_SIZE,
)

neo4j = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

weights = {"doi": 0.1, "paper": 0.3, "abstract": 0.4, "provider": 0.1, "type": 0.1}
client = QdrantClient(url=QDRANT_URI)


def generate_nodes():
    query = "MATCH (n:Paper) RETURN apoc.map.removeKey(properties(n),'embedding') as properties"

    results = neo4j.query(query)

    nodes = []

    for r in results:
        properties = r.get("properties")
        info = properties.get("info")
        abstract = info.split("Abstract: ")[1]
        node = {
            "doi": properties.get("doi"),
            "paper": properties.get("paper"),
            "abstract": abstract,
            "provider": properties.get("provider"),
            "type": properties.get("type"),
        }
        nodes.append(node)

    return nodes


def create_non_weighted_node_vector_store(nodes):
    single_text_nodes = [
        Document(
            page_content=(
                f"DOI: {n.get('doi')} |"
                f"Title: {n.get('paper')} |"
                f"Abstract: {n.get('abstract')} |"
                f"Provider: {n.get('provider')} |"
                f"Type: {n.get('type')}"
            ),
            metadata=n,
        )
        for n in nodes
    ]

    if not client.collection_exists(NON_WEIGHTED_KG_COLLECTION_NAME):
        client.create_collection(
            collection_name=NON_WEIGHTED_KG_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(
            f"Collection '{NON_WEIGHTED_KG_COLLECTION_NAME}' criada com sucesso (Dimensão: {VECTOR_SIZE}, Métrica: Cosine)."
        )

    print("Inserir nós não pesados no banco de dados")

    QdrantVectorStore.from_documents(
        documents=single_text_nodes,
        collection_name=NON_WEIGHTED_KG_COLLECTION_NAME,
        embedding=embedding_model,
        url=QDRANT_URI,
    )


def create_weighted_embeddings(attr_dict, weight_dict):
    final_vector = None

    for attr, text in attr_dict.items():
        if not text:
            continue

        attribute_vector = np.array(embedding_model.embed_query(text))

        weight = weight_dict.get(attr, 0.1)

        weighted_vector = attribute_vector * weight

        if final_vector is None:
            final_vector = weighted_vector
        else:
            final_vector += weighted_vector

    norm = np.linalg.norm(final_vector)
    if norm > 0:
        final_vector = final_vector / norm

    return final_vector.tolist()


def create_weighted_node_vector_store(nodes):
    weighted_attributes_node_vectors = []

    for n in nodes:
        payload = {}
        payload["metadata"] = dict(n)
        payload["page_content"] = (
            f"DOI: {n.get('doi')} |"
            f"Title: {n.get('paper')} |"
            f"Abstract: {n.get('abstract')} |"
            f"Provider: {n.get('provider')} |"
            f"Type: {n.get('type')}"
        )

        vector = PointStruct(
            id=uuid.uuid4(),
            vector=create_weighted_embeddings(n, weights),
            payload=payload,
        )

        weighted_attributes_node_vectors.append(vector)

    print("Nós pesados obtidos com sucesso")

    if not client.collection_exists(WEIGHTED_KG_COLLECTION_NAME):
        client.create_collection(
            collection_name=WEIGHTED_KG_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(
            f"Collection '{WEIGHTED_KG_COLLECTION_NAME}' criada com sucesso (Dimensão: {VECTOR_SIZE}, Métrica: Cosine)."
        )

    print("Inserindo nós pesados na coleção de nós pesados...")
    client.upsert(
        collection_name=WEIGHTED_KG_COLLECTION_NAME,
        points=weighted_attributes_node_vectors,
    )


def create_vector_retrievers_from_kg():
    non_weighted_vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=NON_WEIGHTED_KG_COLLECTION_NAME, embedding=embedding_model
    )
    weighted_vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=WEIGHTED_KG_COLLECTION_NAME, embedding=embedding_model
    )

    non_weighted_retriever = non_weighted_vector_store.as_retriever(
        search_kwargs={"k": 10}
    )
    weighted_retriever = weighted_vector_store.as_retriever(search_kwargs={"k": 10})

    return non_weighted_retriever, weighted_retriever


def assembly_rag_chain(retriever):
    llm = ChatOpenAI(
        model="deepseek-flash",
        temperature=0,
        max_tokens=None,
        timeout=500,
        api_key=DEEPSEEK_API_KEY,
        max_retries=2,
        base_url="https://api.deepseek.com",
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


def initialize_rory(rag_chain):

    print("=" * 60)
    print(" " * 5, "RORY: Research Orchestration and Review sYstem", " " * 5)
    print("=" * 60)

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nRORY: {response}")
