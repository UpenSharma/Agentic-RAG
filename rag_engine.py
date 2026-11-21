"""Agentic RAG engine: LangGraph workflow over Pinecone + Groq + Tavily.

Graph:
    route_question ──► retrieve ──► grade_documents ──► generate
         │                ▲               │
         │                └── rewrite ◄───┤ (no relevant docs, retries left)
         │                                │
         ├──► web_search ◄────────────────┘ (no relevant docs, retries used up)
         │         │
         │         └──► generate
         └──► generate (direct answer, e.g. greetings)
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Literal, TypedDict

# Use the Windows certificate store so HTTPS works behind the corporate proxy.
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ.setdefault("USER_AGENT", "agentic-rag-app")

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration: change these to match your notebook if needed.
# ---------------------------------------------------------------------------
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"
INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "langgraph-agentic-rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
EMBEDDING_DIM = 384
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 4
MAX_REWRITES = 1
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

REQUIRED_KEYS = ["GROQ_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"]


def missing_keys() -> list[str]:
    return [k for k in REQUIRED_KEYS if not os.getenv(k)]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
_embeddings = None
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [ix["name"] for ix in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def get_vectorstore() -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=_pinecone_index(), embedding=get_embeddings(), namespace=NAMESPACE
        )
    return _vectorstore


def knowledge_base_size() -> int:
    stats = _pinecone_index().describe_index_stats()
    ns = stats.namespaces.get(NAMESPACE)
    return ns.vector_count if ns else 0


def ingest_url(url: str) -> int:
    """Load a web page, split it and add the chunks to Pinecone. Returns chunk count."""
    docs = WebBaseLoader(url).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    # Stable IDs so re-ingesting the same page overwrites instead of duplicating.
    ids = [f"{url}#{c.metadata.get('start_index', i)}" for i, c in enumerate(chunks)]
    get_vectorstore().add_documents(chunks, ids=ids)
    return len(chunks)


def ensure_knowledge_base() -> int:
    """Ingest the default source if the namespace is empty. Returns vector count."""
    count = knowledge_base_size()
    if count == 0:
        ingest_url(SOURCE_URL)
        count = knowledge_base_size()
    return count


def get_llm() -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Graph state and structured outputs
# ---------------------------------------------------------------------------
class Source(TypedDict):
    title: str
    url: str
    snippet: str
    origin: str  # "knowledge base" or "web"


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    route: str
    documents: list[Document]
    web_results: list[Source]
    rewrites: int
    answer: str
    sources: list[Source]


class RouteDecision(BaseModel):
    """Where to send the user's question."""

    destination: Literal["vectorstore", "web_search", "direct"] = Field(
        description=(
            "'vectorstore' for questions about LangGraph, agentic RAG, retrieval, "
            "grading, rewriting or building agents; 'web_search' for current events "
            "or topics clearly outside that knowledge base; 'direct' for greetings, "
            "small talk or questions needing no lookup."
        )
    )


class RelevanceGrade(BaseModel):
    """Whether a retrieved document helps answer the question."""

    relevant: Literal["yes", "no"] = Field(description="'yes' if relevant, otherwise 'no'")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def route_question(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You route user questions. The knowledge base contains LangChain's "
                "LangGraph documentation on agentic RAG: retrieval tools, document "
                "grading, question rewriting, conditional edges and building agents.",
            ),
            ("human", "{question}"),
        ]
    )
    decision = (prompt | get_llm().with_structured_output(RouteDecision, method="json_schema")).invoke(
        {"question": state["question"]}
    )
    return {"route": decision.destination, "search_query": state["question"], "rewrites": 0}


def retrieve(state: RAGState) -> RAGState:
    docs = get_vectorstore().similarity_search(state["search_query"], k=TOP_K)
    return {"documents": docs}


def grade_documents(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You grade whether a retrieved document is relevant to a user question. "
                "Answer 'yes' if it contains keywords or meaning that help answer it.",
            ),
            ("human", "Question: {question}\n\nDocument:\n{document}"),
        ]
    )
    grader = prompt | get_llm().with_structured_output(RelevanceGrade, method="json_schema")
    relevant = [
        d
        for d in state.get("documents", [])
        if grader.invoke({"question": state["question"], "document": d.page_content}).relevant
        == "yes"
    ]
    return {"documents": relevant}


def rewrite_question(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the question so it retrieves better results from a vector "
                "store of LangGraph documentation. Return only the rewritten question.",
            ),
            ("human", "{question}"),
        ]
    )
    new_query = (prompt | get_llm()).invoke({"question": state["question"]}).content.strip()
    return {"search_query": new_query, "rewrites": state.get("rewrites", 0) + 1}


def web_search(state: RAGState) -> RAGState:
    response = TavilySearch(max_results=4).invoke({"query": state["question"]})
    results = response.get("results", []) if isinstance(response, dict) else []
    web: list[Source] = [
        {
            "title": r.get("title") or r.get("url", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "origin": "web",
        }
        for r in results
    ]
    return {"web_results": web}


def generate(state: RAGState) -> RAGState:
    sources: list[Source] = [
        {
            "title": d.metadata.get("title", "Knowledge base"),
            "url": d.metadata.get("source", ""),
            "snippet": d.page_content,
            "origin": "knowledge base",
        }
        for d in state.get("documents", [])
    ] + state.get("web_results", [])

    if sources:
        context = "\n\n".join(
            f"[{i}] ({s['origin']}: {s['url']})\n{s['snippet']}" for i, s in enumerate(sources, 1)
        )
        system = (
            "You are a helpful assistant. Answer the question using only the context "
            "below. Cite sources inline using plain square-bracket numbers only, "
            "like [1] or [2]; never use any other citation format. If the context "
            "does not contain the answer, say so plainly.\n\nContext:\n{context}"
        )
    else:
        context = ""
        system = "You are a friendly, concise assistant for an Agentic RAG demo app.{context}"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    answer = (prompt | get_llm()).invoke({"question": state["question"], "context": context})
    return {"answer": _normalize_citations(answer.content), "sources": sources}


def _normalize_citations(text: str) -> str:
    """Turn model-specific markers like 【1†L1-L4】 into plain [1]."""
    return re.sub(r"【(\d+)[^】]*】", r"[\1]", text)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def after_route(state: RAGState) -> str:
    return {"vectorstore": "retrieve", "web_search": "web_search"}.get(state["route"], "generate")


def after_grading(state: RAGState) -> str:
    if state.get("documents"):
        return "generate"
    if state.get("rewrites", 0) < MAX_REWRITES:
        return "rewrite_question"
    return "web_search"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("route_question", route_question)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_question", rewrite_question)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)

    g.add_edge(START, "route_question")
    g.add_conditional_edges("route_question", after_route, ["retrieve", "web_search", "generate"])
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents", after_grading, ["generate", "rewrite_question", "web_search"]
    )
    g.add_edge("rewrite_question", "retrieve")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def describe_step(node: str, update: dict) -> str:
    """Human-readable description of one graph step, for the UI trace."""
    if node == "route_question":
        return {
            "vectorstore": "Routed to the knowledge base (Pinecone)",
            "web_search": "Routed to web search (Tavily)",
            "direct": "Answering directly, no lookup needed",
        }.get(update.get("route", ""), "Routed question")
    if node == "retrieve":
        return f"Retrieved {len(update.get('documents', []))} chunks from Pinecone"
    if node == "grade_documents":
        n = len(update.get("documents", []))
        return f"Graded chunks: {n} relevant" if n else "Graded chunks: none relevant"
    if node == "rewrite_question":
        return f"Rewrote question: “{update.get('search_query', '')}”"
    if node == "web_search":
        return f"Searched the web: {len(update.get('web_results', []))} results"
    if node == "generate":
        return "Generated answer"
    return node


def stream_answer(graph, question: str) -> Iterator[tuple[str, dict]]:
    """Yield (node_name, state_update) for each step of the graph."""
    for chunk in graph.stream({"question": question}, stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update or {}
