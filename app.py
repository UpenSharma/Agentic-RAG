"""Streamlit front end for the Agentic RAG application.

Run with:  streamlit run app.py
"""

import streamlit as st

import rag_engine as engine

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources (built once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model and connecting to Pinecone…")
def load_graph():
    engine.get_vectorstore()  # warm up embeddings + Pinecone connection
    engine.ensure_knowledge_base()
    return engine.build_graph()


@st.cache_data(ttl=60, show_spinner=False)
def kb_size() -> int:
    return engine.knowledge_base_size()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            badge = "📚 Knowledge base" if s["origin"] == "knowledge base" else "🌐 Web"
            title = s["title"] or s["url"]
            st.markdown(f"**[{i}] {title}** · {badge}")
            if s["url"]:
                st.caption(s["url"])
            snippet = s["snippet"].strip().replace("\n", " ")
            st.write(snippet[:400] + ("…" if len(snippet) > 400 else ""))
            if i < len(sources):
                st.divider()


def render_steps(steps: list[str]) -> None:
    if steps:
        with st.expander("Agent steps"):
            for n, step in enumerate(steps, 1):
                st.markdown(f"{n}. {step}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧭 Agentic RAG")
    st.caption(
        "Routes each question, retrieves from Pinecone, grades the chunks, "
        "rewrites the query or falls back to web search, then answers with citations."
    )

    missing = engine.missing_keys()
    if missing:
        st.error("Missing in .env: " + ", ".join(missing))
        st.stop()

    graph = load_graph()

    st.subheader("Knowledge base")
    col1, col2 = st.columns(2)
    col1.metric("Chunks", kb_size())
    col2.metric("Top-k", engine.TOP_K)
    st.caption(f"Index `{engine.INDEX_NAME}` · namespace `{engine.NAMESPACE}`")

    with st.form("ingest", clear_on_submit=True):
        url = st.text_input("Add a web page", placeholder="https://…")
        if st.form_submit_button("Add to knowledge base", use_container_width=True):
            if not url.startswith(("http://", "https://")):
                st.warning("Enter a full URL starting with http:// or https://")
            else:
                with st.spinner("Loading, splitting and embedding…"):
                    try:
                        n = engine.ingest_url(url)
                        kb_size.clear()
                        st.success(f"Added {n} chunks.")
                    except Exception as exc:  # show errors in the UI, not a crash
                        st.error(f"Could not ingest that page: {exc}")

    st.subheader("Models")
    st.caption(f"LLM: `{engine.LLM_MODEL}` (Groq)")
    st.caption(f"Embeddings: `{engine.EMBEDDING_MODEL}`")
    st.caption("Web search: Tavily")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.title("Agentic RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Ask a question to get started. Try one of these:")
    examples = [
        "How does the agentic RAG graph decide whether to rewrite the question?",
        "What does the document grader do?",
        "What are today's top AI news headlines?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_steps(msg.get("steps", []))
            render_sources(msg.get("sources", []))

question = st.chat_input("Ask about LangGraph, agentic RAG, or anything else…")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        steps: list[str] = []
        final: dict = {}
        with st.status("Thinking…", expanded=True) as status:
            try:
                for node, update in engine.stream_answer(graph, question):
                    step = engine.describe_step(node, update)
                    steps.append(step)
                    status.write(step)
                    final.update(update)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Something went wrong", state="error", expanded=True)
                final["answer"] = f"⚠️ Error while answering: `{exc}`"

        answer = final.get("answer", "I couldn't produce an answer.")
        sources = final.get("sources", [])
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps, "sources": sources}
    )
