# Agentic RAG Assistant

A question-answering assistant built with **LangGraph** that decides for itself how to answer each question. It routes each question, retrieves from a **Pinecone** knowledge base, checks whether the retrieved text is actually relevant, rewrites the query or falls back to **Tavily** web search when it isn't, and answers with numbered citations. It uses Groq-hosted LLMs and comes with a **Streamlit** chat interface.

The knowledge base starts with LangChain's [LangGraph agentic RAG guide](https://docs.langchain.com/oss/python/langgraph/agentic-rag). You can add more web pages from the app.

## How it works

```
question
   │
   ▼
route_question ──► direct answer (greetings, general knowledge)
   │
   ├──► retrieve (Pinecone) ──► grade_documents ──► generate answer
   │         ▲                       │
   │         └──── rewrite_question ◄┤  no relevant chunks (1 retry)
   │                                 │
   └──► web_search (Tavily) ◄────────┘  still nothing relevant
             │
             └──► generate answer
```

1. **Route:** the LLM sends the question to the knowledge base, to web search, or straight to an answer.
2. **Retrieve:** the 4 most similar chunks are fetched from Pinecone.
3. **Grade:** each chunk is checked for relevance, and irrelevant ones are dropped.
4. **Rewrite / fall back:** if no chunk is relevant, the question is rewritten and retrieval runs again. If that also fails, Tavily searches the web.
5. **Generate:** the answer is written only from the gathered sources, with inline citations like `[1]`.

## Features

- Chat interface that shows each agent step live while it works
- Expandable list of sources under every answer, labeled as knowledge base or web
- Add any web page to the knowledge base from the sidebar
- Re-adding the same page updates its chunks instead of duplicating them
- Creates the Pinecone index and loads the default source automatically on first run

## Tech stack

| Component | Used for |
|---|---|
| [LangGraph](https://www.langchain.com/langgraph) | Agent workflow (routing, grading, retries) |
| [Groq](https://groq.com) – `openai/gpt-oss-120b` | Routing, grading, rewriting and answering |
| [Pinecone](https://www.pinecone.io) | Vector database (serverless, cosine, 384 dimensions) |
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings, run locally |
| [Tavily](https://tavily.com) | Web search fallback |
| [Streamlit](https://streamlit.io) | Web front end |

## Project structure

```
├── app.py              # Streamlit chat interface
├── rag_engine.py       # LangGraph pipeline, Pinecone ingestion, settings
├── Agentic_RAG.ipynb   # Step-by-step development notebook
├── Requirement.txt     # Pinned Python dependencies
└── .env.example        # Required API keys (copy to .env)
```

## Getting started

### Prerequisites

- Python 3.12
- Free API keys for [Groq](https://console.groq.com/keys), [Pinecone](https://app.pinecone.io) and [Tavily](https://app.tavily.com)

### 1. Clone and install

```bash
git clone https://github.com/UpenSharma/Agentic-RAG.git
cd Agentic-RAG
python -m venv .venv
```

Activate the virtual environment:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

Then install the dependencies:

```bash
pip install -r Requirement.txt
```

The first install takes a while, because `sentence-transformers` pulls in PyTorch.

### 2. Add your API keys

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=...
TAVILY_API_KEY=...
PINECONE_API_KEY=...
```

`.env` is listed in `.gitignore`, so your keys are never committed.

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. On the first run the app downloads the embedding model, creates the Pinecone index `industry-agentic-rag-kb` if it doesn't exist, and loads the default source page.

## Configuration

All settings are at the top of [`rag_engine.py`](rag_engine.py):

| Setting | Default | Meaning |
|---|---|---|
| `SOURCE_URL` | LangGraph agentic RAG guide | Page loaded into an empty knowledge base |
| `INDEX_NAME` | `industry-agentic-rag-kb` | Pinecone index (lowercase letters, digits and hyphens only) |
| `NAMESPACE` | `langgraph-agentic-rag` | Pinecone namespace |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq chat model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Must match the index dimension (384) |
| `TOP_K` | `4` | Chunks retrieved per search |
| `MAX_REWRITES` | `1` | Query rewrites before falling back to web search |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text splitting for ingestion |

If you change `EMBEDDING_MODEL` to one with a different dimension, also change `EMBEDDING_DIM` and use a new `INDEX_NAME`.

## The notebook

`Agentic_RAG.ipynb` builds the pipeline step by step. Its graph differs slightly from the app's: it grades web results as well as knowledge-base chunks, and it uses the smaller `openai/gpt-oss-20b` model. Use the notebook to explore and experiment, and the app to chat with the finished pipeline.

## Troubleshooting

- **`SSLError: CERTIFICATE_VERIFY_FAILED`:** this happens on corporate networks that inspect HTTPS traffic. The app calls `truststore.inject_into_ssl()` so Python trusts the Windows/macOS certificate store. In the notebook, run those two lines (`import truststore` and `truststore.inject_into_ssl()`) at the top of the first cell.
- **`ModuleNotFoundError` in the notebook:** the notebook kernel isn't using the project's `.venv`. Select the `.venv` interpreter as the kernel and restart it.
- **"Missing in .env" in the sidebar:** one of the three API keys isn't set in `.env`.
- **Pinecone dimension error:** the index was created with a different embedding model. Use a new `INDEX_NAME`.
