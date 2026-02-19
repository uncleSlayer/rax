from fastapi import FastAPI, Query

from rax.query import ask

app = FastAPI(title="RAX - RAG API")


@app.get("/ask")
def ask_question(q: str = Query(description="Your question")):
    return ask(q)
