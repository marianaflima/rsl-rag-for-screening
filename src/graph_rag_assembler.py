import os

from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from utils import chunk_documents


VECTOR_SIZE = 768
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

neo4j = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:v1.5",
    base_url="http://localhost:7869",
    dimensions=VECTOR_SIZE,
)

llm = ChatOpenAI(
    model="deepseek-flash",
    temperature=0,
    max_tokens=None,
    timeout=500,
    api_key=DEEPSEEK_API_KEY,
    max_retries=2,
    base_url="https://api.deepseek.com",
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_bm25_retriever() -> BM25Retriever:
    result = neo4j.query("""
    MATCH (n:Paper)
    RETURN n.doi, n.paper, n.provider, n.pub_date, n.info
    """)

    docs = [
        Document(
            page_content=r.get("n.info"),
            metadata={
                "doi": r.get("n.doi"),
                "title": r.get("n.paper"),
                "provider": r.get("n.provider"),
                "pub_date": r.get("n.pub_date"),
            },
        )
        for r in result
    ]

    chunked_docs = chunk_documents(docs)

    retriever = BM25Retriever.from_documents(chunked_docs)
    retriever.k = 5

    return retriever


def assembly_retriever() -> EnsembleRetriever:
    retrieval_query = """
    WITH node AS paper, score

    RETURN
      paper.info
      + '\nAuthors: ' + apoc.text.join(
          [(paper)-[:HAS_AUTHOR]->(a:Author) | a.info], ', ')
      + '\nKeywords: ' + apoc.text.join(
          [(paper)-[:HAS_KEYWORD]->(k:Keyword) | k.info], ', ')
      + '\nTopics: ' + apoc.text.join(
          [(paper)-[:HAS_TOPIC]->(t:Topic) | t.info], ', ')
      + '\nCo-authorships: ' + apoc.text.join(
          [(paper)-[:HAS_AUTHOR]->(a:Author)-[:CO_AUTHOR]-(co:Author)
            | a.info + ' - CO_AUTHOR -> ' + co.info][0..10], '\n')
      + '\nRelated papers: ' + apoc.text.join(
          [(paper)-[:HAS_KEYWORD]->(:Keyword)<-[:HAS_KEYWORD]-(o:Paper)
            WHERE o <> paper | o.info][0..3], '\n---\n')
      AS text,
      score,
      {paper_id: elementId(paper)} AS metadata
    """

    vector_store = Neo4jVector.from_existing_graph(
        embedding=embedding_model,
        node_label="Paper",
        embedding_node_property="embedding",
        index_name="paper_index",
        text_node_properties=["info"],
        retrieval_query=retrieval_query,
    )

    vector_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    bm25_retriever = get_bm25_retriever()

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[
            0.4,
            0.6,
        ],
    )

    return retriever


def assembly_rag_chain(retriever: EnsembleRetriever):

    template = """
        You are an academic records expert. 
        Use ONLY the data provided to answer.
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


def initialize_rory():
    retriever = assembly_retriever()
    rag_chain = assembly_rag_chain(retriever=retriever)

    print("=" * 60)
    print(" " * 5, "RORY: Research Orchestration and Review sYstem", " " * 5)
    print("=" * 60)

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nRORY: {response}")
