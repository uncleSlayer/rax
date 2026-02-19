import neo4j
from openai import OpenAI

from rax.config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    OPENAI_API_KEY,
)
from rax.ingest import build_embed_model

VECTOR_SEARCH_QUERY = """
CALL db.index.vector.queryNodes('document_embedding', $top_k, $embedding)
YIELD node, score
RETURN node.text AS text, node.source_file AS source_file, score
"""

embed_model = build_embed_model()
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    embedding = embed_model.get_text_embedding(question)

    driver = neo4j.GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    with driver.session() as session:
        results = session.run(
            VECTOR_SEARCH_QUERY, top_k=top_k, embedding=embedding
        )
        chunks = [
            {
                "text": record["text"],
                "source_file": record["source_file"],
                "score": record["score"],
            }
            for record in results
        ]
    driver.close()
    return chunks


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "based only on the provided context. If the context doesn't "
                    "contain enough information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def ask(question: str, top_k: int = 5) -> dict:
    chunks = retrieve(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    sources = [
        {"source_file": c["source_file"], "score": c["score"]} for c in chunks
    ]
    return {"question": question, "answer": answer, "sources": sources}
