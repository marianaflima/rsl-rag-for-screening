from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_neo4j.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
    Document,
)
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

import json
import os

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

neo4j = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

print("Montado o Neo4j")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def obtain_papers() -> list[dict]:
    file_path = "data/final/all-records.json"

    with open(file_path) as f:
        data = json.load(f)

    data = data[:10]

    return data


def create_author_nodes(papers: pd.DataFrame, nodes: list[Node]) -> None:

    authors = [
        str(a).strip()
        for a in papers["authors"].explode().unique()
        if pd.notna(a) and a.strip()
    ]

    print("Autores existentes:", len(authors))

    for author in authors:
        if author:
            info = author.lower()
            node = Node(
                id=f"author:{info}",
                type="Author",
                properties={
                    "info": info,
                },
            )
            nodes.append(node)


def create_keyword_nodes(papers: pd.DataFrame, nodes: list[Node]) -> None:
    keywords = set(
        str(k).strip()
        for k in papers["keywords"].explode().unique()
        if pd.notna(k) and k.strip()
    )

    print("Palavras chave existentes:", len(list(keywords)))

    for keyword in keywords:
        if keyword:
            info = keyword.lower()
            node = Node(
                id=f"keyword:{info}",
                type="Keyword",
                properties={
                    "info": info,
                },
            )
            nodes.append(node)


def create_topic_nodes(papers: pd.DataFrame, nodes: list[Node]) -> None:
    topics = set(
        str(t).strip()
        for t in papers["topics"].explode().unique()
        if pd.notna(t) and t.strip()
    )

    print("Tópicos existentes:", len(list(topics)))

    for topic in topics:
        if topic:
            info = topic.lower()
            node = Node(
                id=f"topic:{info}",
                type="Topic",
                properties={
                    "info": info,
                },
            )
            nodes.append(node)


def create_paper_nodes(papers: pd.DataFrame, nodes: list[Node]) -> None:

    for _, row in papers.iterrows():
        doi = row["doi"]
        title = row["title"]
        abstract = str(row["abstract"]).strip() if pd.notna(row["abstract"]) else ""
        provider = row["provider"]
        pub_date = row["pub_date"]
        paper_type = row["article_type"]

        title = title.replace(".", "")
        info = f"DOI: {doi}. Title: {title.lower()}. Abstract: {abstract}"
        paper_properties = {
            "doi": doi,
            "paper": title.lower(),
            "pub_date": pub_date,
            "info": info,
            "provider": provider,
            "type": paper_type,
        }

        p_node = Node(
            id=f"paper:{doi}",
            type="Paper",
            properties=paper_properties,
        )
        nodes.append(p_node)


def create_nodes(data: list[dict]) -> tuple[list[Node],]:

    papers = pd.DataFrame(data=data)

    nodes = list[Node]()

    print("=" * 60)
    print(" " * 10, "CREATE GRAPH", " " * 10)
    print("=" * 60)
    print("Artigos existentes:", len(data))

    create_author_nodes(papers=papers, nodes=nodes)
    create_topic_nodes(papers=papers, nodes=nodes)
    create_keyword_nodes(papers=papers, nodes=nodes)
    create_paper_nodes(papers=papers, nodes=nodes)

    author_index, topic_index, keyword_index, paper_index = create_node_indexes(nodes)

    print("=" * 60)
    print(f"Total de nós criados: {len(nodes)}")
    print("=" * 60)
    print(f"- Autores: {len(author_index)}")
    print(f"- Tópicos: {len(topic_index)}")
    print(f"- Keywords: {len(keyword_index)}")
    print(f"- Artigos: {len(paper_index)}")
    print("=" * 60)

    return nodes


def create_node_indexes(nodes: list[Node]) -> tuple[dict, dict, dict, dict]:
    author_index = {n.id: n for n in nodes if n.type == "Author"}
    topic_index = {n.id: n for n in nodes if n.type == "Topic"}
    keyword_index = {n.id: n for n in nodes if n.type == "Keyword"}
    paper_index = {n.id: n for n in nodes if n.type == "Paper"}

    return author_index, topic_index, keyword_index, paper_index


def create_relationships(data: list[dict], nodes: list[Node]) -> list[Relationship]:
    relationships = list[Relationship]()

    papers = pd.DataFrame(data)

    author_index, topic_index, keyword_index, paper_index = create_node_indexes(nodes)

    for _, row in papers.iterrows():
        doi = row["doi"]
        p_node = paper_index.get(f"paper:{doi}")

        row_authors = row.get("authors", [])
        if isinstance(row_authors, list):
            for a in row_authors:
                a_clean = str(a).strip().lower()
                a_node = author_index.get(f"author:{a_clean}")
                if a_node is not None:
                    relationships.append(
                        Relationship(source=p_node, type="HAS_AUTHOR", target=a_node)
                    )

        # Relacionamentos com Tópicos
        row_topics = row.get("topics", [])
        if isinstance(row_topics, list):
            for t in row_topics:
                t_clean = str(t).strip().lower()
                t_node = topic_index.get(f"topic:{t_clean}")
                if t_node is not None:
                    relationships.append(
                        Relationship(source=p_node, type="HAS_TOPIC", target=t_node)
                    )

        # Relacionamentos com Keywords
        row_keywords = row.get("keywords", [])
        if isinstance(row_keywords, list):
            for k in row_keywords:
                k_clean = str(k).strip().lower()
                k_node = keyword_index.get(f"keyword:{k_clean}")
                if k_node is not None:
                    relationships.append(
                        Relationship(source=p_node, type="HAS_KEYWORD", target=k_node)
                    )

        co_author_nodes = [
            r.target
            for r in relationships
            if r.type == "HAS_AUTHOR" and r.source.id == f"paper:{doi}"
        ]

        for author in co_author_nodes:
            for co_author in co_author_nodes:
                if author.id != co_author.id:
                    r = Relationship(source=author, type="CO_AUTHOR", target=co_author)
                    if r not in relationships:
                        relationships.append(r)

    has_topic_rel = [r for r in relationships if r.type == "HAS_TOPIC"]
    has_keyword_rel = [r for r in relationships if r.type == "HAS_KEYWORD"]
    has_author_rel = [r for r in relationships if r.type == "HAS_AUTHOR"]
    co_author_rel = [r for r in relationships if r.type == "CO_AUTHOR"]

    print(f"Total de relacionamentos criados: {len(relationships)}")
    print("=" * 60)
    print(f"- HAS_AUTHOR: {len(has_author_rel)}")
    print(f"- HAS_TOPIC: {len(has_topic_rel)}")
    print(f"- HAS_KEYWORD: {len(has_keyword_rel)}")
    print(f"- CO_AUTHOR: {len(co_author_rel)}")
    print("=" * 60)

    return relationships


def create_graph() -> None:
    data = obtain_papers()

    nodes = create_nodes(data=data)
    relationships = create_relationships(data=data, nodes=nodes)

    graph_document = GraphDocument(nodes=nodes, relationships=relationships)

    neo4j.query("MATCH (n) DETACH DELETE n")
    neo4j.add_graph_documents(graph_documents=[graph_document], baseEntityLabel=True)
