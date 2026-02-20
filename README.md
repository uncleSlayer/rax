# RAX - RAG API

A Retrieval-Augmented Generation API that combines semantic search over PDF documents with LLM-powered answer generation using Neo4j, OpenAI, and FastAPI.

## How It Works

1. **Ingest** — PDFs in `/data` are semantically chunked, embedded via OpenAI (`text-embedding-3-small`), and stored in Neo4j as vectors.
2. **Query** — User questions are embedded and matched against stored chunks using cosine similarity. The top results are fed as context to GPT-4o-mini, which generates an answer with source attribution.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker & Docker Compose (for Neo4j)
- OpenAI API key

## Setup

```bash
# Install dependencies
uv sync

# Start Neo4j
docker-compose up -d

# Configure environment
cp .env.example .env  # then add your OPENAI_API_KEY
```

## Usage

### Ingest documents

Place PDF files in the `/data` directory, then run:

```python
from rax.ingest import run
run()
```

### Start the server

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### Query

```bash
curl "http://localhost:8000/ask?q=your+question+here"
```

**Response:**

```json
{
  "question": "your question here",
  "answer": "...",
  "sources": ["filename.pdf"]
}
```

## Project Structure

```
rax/
├── rax/
│   ├── config.py      # Environment and model configuration
│   ├── ingest.py      # PDF loading, chunking, embedding, and storage
│   ├── query.py       # Vector retrieval and LLM answer generation
│   └── server.py      # FastAPI application
├── data/              # PDF documents to index
├── main.py            # Entry point (runs uvicorn)
├── docker-compose.yml # Neo4j service
└── pyproject.toml     # Dependencies and project metadata
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector DB | Neo4j 5.0 |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | GPT-4o-mini |
| API | FastAPI + Uvicorn |
| Document Processing | LlamaIndex (semantic splitting) |

## License

MIT
