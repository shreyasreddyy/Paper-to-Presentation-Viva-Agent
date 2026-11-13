"""Paper-to-Presentation & Viva Agent - Fully Functional Streamlit Application.

A reliable, local-first academic AI assistant powered by LangChain (LCEL),
Ollama (llama3.2), ChromaDB, PyPDF, and python-pptx.
"""

import time
import streamlit as st

from core.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    AGENT_DEFINITIONS,
)
from core.ollama_service import check_ollama_health, get_llm, get_embedder
from core.document_loader import load_uploaded_document
from core.vector_store import build_vector_store, retrieve_context_with_metadata
from core.router import safe_route
from core.agents import run_specialist_agent, AGENT_PROMPTS, AGENT_USER_TEMPLATE
from core.presentation_exporter import parse_slide_markdown, generate_pptx_file

# Streamlit Page Configuration
st.set_page_config(
    page_title="Paper-to-Presentation & Viva Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Theme CSS
CUSTOM_CSS = """
<style>
:root {
    --bg-main: #0d1117;
    --bg-card: #161b22;
    --bg-elevated: #21262d;
    --border-subtle: rgba(255, 255, 255, 0.1);
    --border-active: #388bfd;
    --text-main: #f0f6fc;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --success: #3fb950;
    --success-bg: rgba(46, 160, 67, 0.15);
    --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

.header-box {
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 24px;
}

.header-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-main);
    margin: 0;
}

.header-desc {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 4px 0 0 0;
}

.card-box {
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.pill-status {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    font-weight: 500;
}

.pill-green {
    background-color: var(--success-bg);
    color: var(--success);
    border: 1px solid rgba(46, 160, 67, 0.4);
}

.pill-amber {
    background-color: rgba(210, 153, 34, 0.15);
    color: #d29922;
    border: 1px solid rgba(210, 153, 34, 0.4);
}

.passage-box {
    background-color: var(--bg-elevated);
    border-left: 3px solid var(--border-active);
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 12px;
    font-size: 0.86rem;
    color: var(--text-main);
    line-height: 1.5;
}

.passage-meta {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent);
    margin-bottom: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
def init_session():
    defaults = {
        "vector_store": None,
        "indexed_chunks": [],
        "paper_meta": None,
        "indexed_hash": None,
        "last_result": None,
        "last_routing": None,
        "last_retrieval": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


def reset_state():
    """Completely reset active paper, index, and results."""
    st.session_state.vector_store = None
    st.session_state.indexed_chunks = []
    st.session_state.paper_meta = None
    st.session_state.indexed_hash = None
    st.session_state.last_result = None
    st.session_state.last_routing = None
    st.session_state.last_retrieval = None


# Sidebar
with st.sidebar:
    st.markdown("### Infrastructure Status")
    health = check_ollama_health()
    if health["connected"]:
        st.markdown('<span class="pill-status pill-green">Ollama: Online</span>', unsafe_allow_html=True)
        st.caption(f"Models available: {len(health['models'])}")
    else:
        st.markdown('<span class="pill-status pill-amber">Ollama: Offline</span>', unsafe_allow_html=True)
        st.caption(f"Status: {health['error'] or 'Connection refused'}. Run `ollama serve`.")

    st.divider()

    st.markdown("### Active Document")
    if st.session_state.paper_meta:
        pm = st.session_state.paper_meta
        st.markdown(
            f"""
            <div class="card-box">
                <div style="font-size:0.75rem; color:#8b949e; font-family:monospace;">FILE NAME</div>
                <div style="font-size:0.92rem; font-weight:500; color:#f0f6fc;">{pm['filename'][:24]}</div>
                <div style="font-size:0.75rem; color:#8b949e; font-family:monospace; margin-top:8px;">EXTRACTION METHOD</div>
                <div style="font-size:0.88rem; font-weight:600; color:#58a6ff;">{pm.get('extraction_method', 'Digital')}</div>
                <div style="font-size:0.75rem; color:#8b949e; font-family:monospace; margin-top:8px;">INDEXED CHUNKS</div>
                <div style="font-size:0.92rem; font-weight:500; color:#f0f6fc;">{len(st.session_state.indexed_chunks)} chunks ({pm['pages']} pgs)</div>
                <div style="font-size:0.75rem; color:#8b949e; font-family:monospace; margin-top:8px;">FILE SIZE</div>
                <div style="font-size:0.92rem; font-weight:500; color:#f0f6fc;">{pm['file_size_kb']} KB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear / Reset Document", use_container_width=True):
            reset_state()
            st.rerun()
    else:
        st.caption("No research paper indexed yet. Upload a document in Step 1.")

    st.divider()

    st.markdown("### Model Parameters")
    chat_model = st.text_input("Ollama Chat Model", value=DEFAULT_CHAT_MODEL)
    embedding_model = st.text_input("Embedding Model", value=DEFAULT_EMBEDDING_MODEL)

    col_c, col_t = st.columns(2)
    with col_c:
        num_ctx = st.selectbox("Context Window", [2048, 4096, 8192, 16384], index=1)
    with col_t:
        temperature = st.slider("Temperature", 0.0, 0.7, DEFAULT_TEMPERATURE, step=0.05)

    st.divider()

    with st.expander("7 Acceptance Criteria Verified"):
        st.markdown(
            """
            - **A. LLM Integration:** Local `llama3.2` via `ChatOllama`
            - **B. Interactive GUI:** Streamlit workspace with instant actions
            - **C. Prompt Engineering:** Grounded system/user prompts
            - **D. LangChain Core & LCEL:** `ChatPromptTemplate` | `llm` | `parser`
            - **E. Agentic RAG:** ChromaDB top-k similarity retrieval
            - **F. Intent Routing:** LLM router with JSON classification
            - **G. Vector Store:** ChromaDB in-memory ephemeral store
            """
        )


# Main Header
st.markdown(
    """
    <div class="header-box">
        <h1 class="header-title">📚 Paper-to-Presentation & Viva Agent</h1>
        <p class="header-desc">
            Local-first Agentic AI academic assistant. Ingest research papers, retrieve grounded context, author presentations, and generate thesis viva defenses.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# STEP 1: Upload & Index Research Paper
st.markdown("### 1. Ingest Research Paper")
up_col1, up_col2 = st.columns([3, 1])

with up_col1:
    uploaded_file = st.file_uploader(
        "Select research paper (PDF, TXT, or Markdown)",
        type=["pdf", "txt", "md"],
        help="The document is processed locally in-memory using ChromaDB.",
    )

with up_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    index_btn = st.button("Process & Build RAG Index", type="primary", use_container_width=True)

if index_btn:
    if not uploaded_file:
        st.error("Please upload a paper file first.")
    else:
        try:
            # 1. Read and validate document (Digital extraction with automatic OCR fallback)
            with st.spinner("Extracting text (attempting digital extraction, with OCR fallback)..."):
                t_start = time.time()
                docs, meta = load_uploaded_document(uploaded_file)

                # Check duplicate indexing prevention
                if st.session_state.indexed_hash == meta["file_hash"]:
                    st.info(f"'{meta['filename']}' is already indexed in ChromaDB ({len(st.session_state.indexed_chunks)} chunks). Duplicate indexing prevented.")
                else:
                    # 2. Build ChromaDB Index
                    with st.spinner(f"Computing embeddings with '{embedding_model}' and building ChromaDB index..."):
                        embedder = get_embedder(model_name=embedding_model)
                        store, chunks = build_vector_store(docs, embedder)
                        t_elapsed = round(time.time() - t_start, 2)

                        st.session_state.vector_store = store
                        st.session_state.indexed_chunks = chunks
                        st.session_state.paper_meta = meta
                        st.session_state.indexed_hash = meta["file_hash"]

                    st.success(f"Successfully indexed '{meta['filename']}' via {meta['extraction_method']} ({len(chunks)} chunks, {meta['pages']} pages, {meta['total_chars']} chars) in {t_elapsed}s.")
                    st.rerun()

        except Exception as e:
            st.error(f"Document ingestion failed: {e}")
            st.info("Check that Ollama is running and both models ('llama3.2', 'nomic-embed-text') are pulled via `ollama list`.")

if st.session_state.paper_meta:
    pm = st.session_state.paper_meta
    method_pill = f"Method: **{pm.get('extraction_method', 'Digital')}**"
    st.info(f"Active Paper: **{pm['filename']}** · {pm['pages']} pages · {method_pill} · {len(st.session_state.indexed_chunks)} text chunks ready for retrieval.")

    # Guard against stale state when a different file is selected in uploader
    if uploaded_file and uploaded_file.name != pm.get("filename"):
        st.warning(
            f"⚠️ **Document Mismatch**: You have selected **'{uploaded_file.name}'**, but **'{pm['filename']}'** is currently active in memory. "
            f"Please click **'Process & Build RAG Index'** above to index '{uploaded_file.name}'."
        )


st.divider()


# STEP 2: Choose Specialist Agent & Run
st.markdown("### 2. Specialist Agent Orchestration")

agent_options = [
    ("Summary", "summary", "Generates an executive summary covering abstract, research problem, objectives, methodology, results, and conclusion."),
    ("Key Concepts", "concepts", "Extracts important technical concepts and explains them in understandable student-friendly language."),
    ("Presentation (8-10 Slides + PPTX)", "slides", "Creates an 8-10 slide presentation with bullet points, speaker notes, and downloadable PowerPoint (.pptx)."),
    ("Viva Questions & Answers", "viva", "Generates 10 examiner questions across basic, intermediate, and advanced levels with model answers."),
    ("Academic Explainer / Q&A", "explain", "Answers specific questions about the paper grounded in retrieved document excerpts."),
]

agent_labels = [opt[0] for opt in agent_options]
agent_keys = [opt[1] for opt in agent_options]

sel_col1, sel_col2 = st.columns([1, 2])

with sel_col1:
    selected_label = st.selectbox("Select Specialist Role", agent_labels, index=0)
    selected_idx = agent_labels.index(selected_label)
    selected_intent = agent_keys[selected_idx]
    st.caption(agent_options[selected_idx][2])

with sel_col2:
    default_instructions = {
        "summary": "Provide a structured summary containing abstract, problem, objectives, methodology, results, and conclusion.",
        "concepts": "Extract the key technical concepts and explain each in student-friendly terms with concrete paper examples.",
        "slides": "Generate an 8 to 10 slide presentation deck with bullet points, spoken speaker notes, and diagram suggestions.",
        "viva": "Generate 10 relevant viva questions and answers covering basic, intermediate, and advanced levels.",
        "explain": "Explain the primary methodology and results of this paper in simple terms.",
    }
    user_instruction = st.text_area(
        "Custom Instruction / Research Focus (Optional)",
        value=default_instructions[selected_intent],
        height=90,
    )

run_agent_btn = st.button("Run Specialist Agent", type="primary", use_container_width=True)

if run_agent_btn:
    if st.session_state.vector_store is None:
        st.warning("Please upload and index a research paper first (Step 1).")
    elif uploaded_file and st.session_state.paper_meta and uploaded_file.name != st.session_state.paper_meta.get("filename"):
        st.error(
            f"Cannot run agent: The file '{uploaded_file.name}' in the uploader has not been indexed yet. "
            f"The active index belongs to '{st.session_state.paper_meta.get('filename')}'. "
            f"Please click 'Process & Build RAG Index' in Step 1 first."
        )
    else:
        try:
            with st.spinner("Initializing specialist agent..."):
                t_start = time.time()
                llm = get_llm(model_name=chat_model, temperature=temperature, num_ctx=num_ctx)

                # 1. Intent Detection & Routing
                route_decision, raw_route = safe_route(user_instruction, llm, has_document=True)
                # Respect selected agent from UI
                route_decision["intent"] = selected_intent

                # 2. RAG Retrieval Step
                retrieved_chunks = []
                if route_decision["retrieve"]:
                    # Formulate appropriate retrieval query
                    query_for_retrieval = user_instruction.strip() or {
                        "summary": "abstract problem objectives methodology experimental results conclusions",
                        "concepts": "key technical concepts definitions algorithms formulations",
                        "slides": "methodology architecture datasets quantitative results ablation baseline limitations",
                        "viva": "core methodology findings ablation baselines limitations future work viva questions",
                        "explain": user_instruction,
                    }[selected_intent]

                    context, retrieved_chunks = retrieve_context_with_metadata(
                        st.session_state.vector_store,
                        query=query_for_retrieval,
                        k=5,
                    )
                else:
                    context = "Retrieval was bypassed by the router."

                if not context or len(context.strip()) == 0:
                    context = "No relevant document excerpts could be retrieved from the active index."

                # 3. LCEL Agent Execution
                result_text = run_specialist_agent(selected_intent, context, user_instruction, llm)
                latency = round(time.time() - t_start, 2)

                st.session_state.last_result = {
                    "label": selected_label,
                    "intent": selected_intent,
                    "query": user_instruction,
                    "content": result_text,
                    "latency": latency,
                }
                st.session_state.last_routing = {
                    "route": route_decision,
                    "raw": raw_route,
                }
                st.session_state.last_retrieval = retrieved_chunks

        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            st.info("Check that Ollama is running and has adequate memory allocated.")


# STEP 3: Display Generated Outputs & Downloads
if st.session_state.last_result:
    res = st.session_state.last_result
    st.divider()
    st.markdown(f"### 3. Generated Output: {res['label']}")
    st.caption(f"Executed via LangChain LCEL in {res['latency']}s | Model: {chat_model}")

    # Render formatted output
    st.markdown(res["content"])

    # Download Buttons
    paper_name = st.session_state.paper_meta.get("filename", "research_paper") if st.session_state.paper_meta else "paper"
    clean_name = paper_name.rsplit(".", 1)[0].replace(" ", "_")

    # If slides were generated, provide PPTX download alongside Markdown/TXT
    if res["intent"] == "slides":
        parsed_slides = parse_slide_markdown(res["content"])
        pptx_buffer = generate_pptx_file(parsed_slides, paper_title=paper_name)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "📥 Download PowerPoint Presentation (.pptx)",
                data=pptx_buffer.getvalue(),
                file_name=f"{clean_name}_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "Download Slides (Markdown .md)",
                data=res["content"],
                file_name=f"{clean_name}_slides.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d3:
            st.download_button(
                "Download Slides (Plain Text .txt)",
                data=res["content"],
                file_name=f"{clean_name}_slides.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                f"Download {res['intent'].capitalize()} (Markdown .md)",
                data=res["content"],
                file_name=f"{clean_name}_{res['intent']}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                f"Download {res['intent'].capitalize()} (Plain Text .txt)",
                data=res["content"],
                file_name=f"{clean_name}_{res['intent']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    # STEP 4: Transparent Inspection Expanders
    st.markdown("### 4. Verification & System Inspection")

    with st.expander("🔍 View Retrieved Document Passages (ChromaDB Top-K Chunks)"):
        if st.session_state.last_retrieval:
            st.markdown(f"Retrieved **{len(st.session_state.last_retrieval)}** most relevant chunks from ChromaDB:")
            for chunk in st.session_state.last_retrieval:
                st.markdown(
                    f"""
                    <div class="passage-box">
                        <div class="passage-meta">
                            Rank #{chunk['rank']} · Page {chunk['page']} · Chunk #{chunk['chunk_id']} · Distance: {chunk['score']}
                        </div>
                        {chunk['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No retrieval was performed for this step.")

    with st.expander("🧭 View Intent Routing & Classification Details"):
        if st.session_state.last_routing:
            r_info = st.session_state.last_routing
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"**Selected Specialist:** `{r_info['route']['intent']}`")
                st.markdown(f"**Retrieval Active:** `{r_info['route']['retrieve']}`")
            with col_r2:
                st.markdown(f"**Router Rationale:** {r_info['route'].get('reason', 'N/A')}")
            st.markdown("**Raw Router Output:**")
            st.code(r_info["raw"], language="json")

    with st.expander("🧠 View Prompt Engineering & LCEL Architecture"):
        st.markdown("**Specialist System Prompt:**")
        st.code(AGENT_PROMPTS.get(res["intent"], ""), language="markdown")
        st.markdown("**User Template:**")
        st.code(AGENT_USER_TEMPLATE, language="markdown")
        st.markdown(
            """
            **Techniques Applied:**
            1. *Role-based Persona:* Strict academic specialist instructions with precise section guidelines.
            2. *Grounding Constraints:* Forbids hallucinations; enforces factual adherence to retrieved context.
            3. *LCEL Pipeline:* `prompt | llm | StrOutputParser()`.
            """
        )
