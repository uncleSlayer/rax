from pathlib import Path

import neo4j
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

from rax.config import (
    CHUNK_BREAKPOINT_PERCENTILE,
    CHUNK_BUFFER_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    OPENAI_API_KEY,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CREATE_DOCUMENT_QUERY = """
UNWIND $documents AS doc
CREATE (d:Document {
    text: doc.text,
    embedding: doc.embedding,
    source_file: doc.source_file,
    chunk_index: doc.chunk_index
})
"""

CREATE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX document_embedding IF NOT EXISTS
FOR (d:Document)
ON (d.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: $dimensions,
        `vector.similarity_function`: 'cosine'
    }
}
"""


def build_embed_model() -> OpenAIEmbedding:
    return OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def load_and_chunk(embed_model: OpenAIEmbedding) -> list:
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.pdf")):
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    print(f"Loading PDFs from {DATA_DIR}")
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR), required_exts=[".pdf"]
    ).load_data()
    print(f"Loaded {len(documents)} document pages")

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=CHUNK_BREAKPOINT_PERCENTILE,
        buffer_size=CHUNK_BUFFER_SIZE,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} semantic chunks")
    return nodes


def embed_nodes(nodes: list, embed_model: OpenAIEmbedding) -> list[dict]:
    texts = [node.get_content() for node in nodes]
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    documents = []
    for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
        source_file = node.metadata.get("file_name", "unknown")
        documents.append(
            {
                "text": node.get_content(),
                "embedding": embedding,
                "source_file": source_file,
                "chunk_index": i,
            }
        )
    return documents


def store_in_neo4j(documents: list[dict]) -> None:
    driver = neo4j.GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    with driver.session() as session:
        session.run(
            CREATE_VECTOR_INDEX_QUERY, dimensions=EMBEDDING_DIMENSIONS
        )
        print("Created vector index")

        # Insert in batches of 100
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            session.run(CREATE_DOCUMENT_QUERY, documents=batch)
            print(f"Stored batch {i // batch_size + 1} ({len(batch)} chunks)")

    driver.close()
    print(f"Stored {len(documents)} chunks in Neo4j")


def run() -> None:
    embed_model = build_embed_model()
    nodes = load_and_chunk(embed_model)
    documents = embed_nodes(nodes, embed_model)
    store_in_neo4j(documents)
    print("Ingestion complete!")
