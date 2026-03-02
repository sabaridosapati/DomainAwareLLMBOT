import streamlit as st
import uuid
import warnings
from chatbot import DomainAwareChatbot

warnings.filterwarnings(
    "ignore",
    message="Importing verbose from langchain root module is no longer supported.*",
)

st.set_page_config(
    page_title="Domain-Aware AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Init ────────────────────────────────────────────────
if "chatbot" not in st.session_state:
    st.session_state.chatbot = DomainAwareChatbot()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ──────────────────────────────────────────────
st.title("🤖 Domain-Aware LLM Chatbot")
st.caption("Automatically routes your query to the best AI model for the domain")

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    use_llm_classifier = st.toggle("Use LLM Classifier (more accurate)", value=False)
    st.markdown("---")
    st.markdown("**Domain → Model Routing:**")
    st.markdown("🏥 Medical → gemini-2.5-pro")
    st.markdown("⚖️ Legal → gemini-2.5-pro")
    st.markdown("💰 Finance → gemini-2.5-flash")
    st.markdown("💻 Technology → Mistral")
    st.markdown("💬 General → Mistral")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chatbot.memory_mgr.clear(st.session_state.session_id)
        st.rerun()

# ── Domain Badge Colors ──────────────────────────────────
DOMAIN_COLORS = {
    "medical":    "🏥 Medical",
    "legal":      "⚖️ Legal",
    "finance":    "💰 Finance",
    "technology": "💻 Technology",
    "general":    "💬 General"
}

# ── Chat History ────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].caption(f"🎯 Domain: {DOMAIN_COLORS.get(meta['domain'], meta['domain'])}")
            cols[1].caption(f"🤖 Model: {meta['model_used']}")
            cols[2].caption(f"📚 Chunks: {meta['context_chunks']}")

            if meta.get("sources"):
                with st.expander("📎 Sources"):
                    for s in meta["sources"]:
                        st.write(f"- {s['source']} (Page {s['page']})")

# ── Chat Input ───────────────────────────────────────────
if prompt := st.chat_input("Ask anything — medical, legal, finance, tech..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Routing to best model..."):
            result = st.session_state.chatbot.chat(
                query=prompt,
                session_id=st.session_state.session_id,
                use_llm_classifier=use_llm_classifier
            )

        st.write(result["answer"])

        cols = st.columns(3)
        cols[0].caption(f"🎯 {DOMAIN_COLORS.get(result['domain'], result['domain'])}")
        cols[1].caption(f"🤖 {result['model_used']}")
        cols[2].caption(f"📚 {result['context_chunks']} chunks")

        if result.get("sources"):
            with st.expander("📎 Sources"):
                for s in result["sources"]:
                    st.write(f"- {s['source']} (Page {s['page']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": result
    })
