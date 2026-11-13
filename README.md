# Paper-to-Presentation & Viva Agent

A local-first, privacy-preserving Agentic AI research assistant and thesis defense workspace designed for academic synthesis, conference slide authoring, and oral defense viva preparation.

Built with **Python 3.11**, **Streamlit**, **LangChain (LCEL)**, **Ollama (`llama3.2`)**, **ChromaDB**, **PyPDF**, and **python-pptx**.

---

## 1. System Architecture

The application implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline orchestrated entirely on local infrastructure with zero cloud dependencies or paid APIs.

```
                   User Upload (PDF / TXT / MD - Multi-Document Support)
                                │
                                ▼
                       Document Ingestion
                    (PyPDFLoader / TextLoader)
                                │
                                ▼
                        Semantic Chunking
          RecursiveCharacterTextSplitter (900 chars, 150 overlap)
                                │
                                ▼
                        Vector Embedding
              OllamaEmbeddings (nomic-embed-text)
                                │
                                ▼
                   ChromaDB Ephemeral Store
                                │
       ┌────────────────────────┴────────────────────────┐
       │                                                 │
       ▼                                                 ▼
User Specialist Query                            Interactive Defense Chat
       │                                                 │
       ▼                                                 ▼
Intent Router (safe_route)                       ChromaDB Retrieval
(ChatPromptTemplate | ChatOllama | JSON Schema)          │
       │                                                 ▼
  ┌────┴────────────────────────┐                Academic Explainer
  │                             │                        │
  ▼                             ▼                        ▼
retrieve = True           retrieve = False        Grounded Answer
  │                             │                 (with Citations & Pages)
  ▼                             │
ChromaDB Similarity Search      │
(top-k chunks + distance scores)│
  │                             │
  └──────────────┬──────────────┘
                 │
                 ▼
       Specialist Agent Team
 (ChatPromptTemplate | ChatOllama | StrOutputParser)
   ├── 1. Summarizer Agent (Executive Research Digest)
   ├── 2. Concept-Extractor Agent (Foundations & Intuitions)
   ├── 3. Slide-Generator Agent (8-12 Slides + PPTX Download)
   ├── 4. Viva Agent (6 Tiers + Oral Exam Practice Simulator)
   └── 5. Academic Explainer Agent (Grounded Defense Inquiry)
                 │
                 ▼
       Academic SaaS Workspace
   ├── Dashboard & Live Infrastructure Telemetry
   ├── Research Studio (Multi-File Ingestion & Specialist Agents)
   ├── Presentation Studio (Slide Deck & PowerPoint .pptx Export)
   ├── Viva Examination Studio (Syllabus & Interactive Oral Defense Simulator)
   ├── AI Research Chat (Conversational Grounded Q&A with Citations)
   ├── Document Explorer (Keyword Filter & Semantic Retrieval Sandbox)
   └── Architecture & Diagnostics (Pipeline Flow & Criteria Scorecard)
```

---

## 2. Seven Mandatory Acceptance Criteria Mapping

| # | Acceptance Criterion | Technical Implementation in Codebase |
|---|---|---|
| **A** | **LLM Integration** | Integrates local `llama3.2` via `langchain_ollama.ChatOllama` with bounded context window (`num_ctx=4096`) for low-latency CPU/GPU execution. Includes live Ollama connection health monitoring. |
| **B** | **Interactive GUI** | Modern, dark-themed Streamlit workspace with 7 dedicated views: Dashboard, Research Studio, Presentation Studio, Viva Examination Studio, AI Research Chat, Document Explorer, and Architecture Diagnostics. |
| **C** | **Prompt Engineering** | Strict separation of system prompts and user prompts. Employs role-based personas, few-shot JSON formatting schema, and strict grounding constraints forbidding factual hallucinations. |
| **D** | **LangChain Core & LCEL** | Composed with `ChatPromptTemplate.from_messages([('system', ...), ('human', ...)])` and compiled via the LangChain Expression Language pipe syntax (`prompt \| llm \| StrOutputParser()`). |
| **E** | **Agentic RAG** | ChromaDB semantic similarity search retrieves top-k passages with metadata (page numbers, chunk IDs, and distance scores). The intent router dynamically decides if retrieval is necessary. |
| **F** | **Intent Detection & Routing** | `safe_route()` employs an LLM classifier with few-shot schema enforcement to categorize intent (`summary`, `concepts`, `slides`, `viva`, `explain`) with deterministic fallback and document-presence gating. |
| **G** | **Vector Database** | In-memory ChromaDB vector store powered by `nomic-embed-text` embeddings, isolated per session to prevent cross-document leakage and file-locking issues on Windows. |

---

## 3. Specialist Agent Team

### 1. Summarizer Agent
Generates an exhaustive, structured executive summary:
- Background & Research Motivation
- Core Problem Formulation
- Research Objectives & Novel Contributions
- Methodology & Architectural Overview
- Experimental Evaluation & Key Metrics
- Limitations & Threats to Validity
- Conclusion & Research Implications

### 2. Concept-Extractor Agent
Extracts 4 to 6 foundational concepts with pedagogical clarity:
- Academic Formal Definition
- Student-Friendly Intuition (analogies in plain English)
- Technical & Mathematical Formulations
- Application in This Paper
- Architectural Significance

### 3. Slide-Generator Agent (Presentation Studio)
Creates an 8 to 12 slide conference or defense presentation:
- Structured progression (Problem, Related Work, Method, Setup, Results, Limitations, Conclusion)
- 3 to 5 concise bullet points per slide
- Spoken **Speaker Notes** for the presenter during the defense
- Visual / Diagram Recommendation for each slide
- **Direct Downloadable PowerPoint (`.pptx`)** output generated via `python-pptx` with embedded notes!

### 4. Viva Agent (Viva Examination Studio)
Prepares students for thesis defense examinations across 6 tiers:
- **Tier 1:** Basic & Foundational Concepts
- **Tier 2:** Problem Domain & Literature Justification
- **Tier 3:** Methodology, Algorithms & Architecture
- **Tier 4:** Experimental Setup, Datasets & Metrics
- **Tier 5:** Results, Ablation Studies & Validation
- **Tier 6:** Critical Defense, Limitations & Future Scope
- **Interactive Oral Defense Practice Simulator**: Evaluates student typed answers in real-time with an AI examiner score (0-10), constructive feedback, omitted concepts, and improved delivery tips.

### 5. Academic Explainer Agent (AI Research Chat)
Provides interactive, cited explanations for specific user inquiries with strict information boundaries and document page numbers (`[Page X, Chunk Y]`).

---

## 4. Setup and Installation (Windows)

### Prerequisites
- **Python 3.11**
- **Ollama** installed and running ([Download Ollama](https://ollama.com))

### 1. Pull Required Local Models
Open Command Prompt or PowerShell:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify that models are available:
```bash
ollama list
```

### 2. Activate Virtual Environment
In the project directory `D:\paper_viva_agent`:
```bash
.\venv311\Scripts\activate
```

### 3. Run Automated Acceptance Tests
```bash
python test_system.py
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 5. College Project Demonstration Script

Follow these steps for a live project demonstration:

1. **Dashboard & Telemetry**:
   - Show the live infrastructure KPI cards (Ollama connected, `llama3.2` active, `nomic-embed-text` active).
   - Point out that processing runs 100% locally on the host machine without external API keys or cloud subscriptions.

2. **Multi-Document Ingestion**:
   - In **Research Studio**, upload a research paper (`.pdf`, `.txt`, or `.md`).
   - Click **Process & Build RAG Index**.
   - Show the real-time progress bar and indexing summary (document count, chunk count, and character volume).

3. **Specialist Agent Execution**:
   - Select **Concept Extractor Agent** and click **Run Specialist Agent**.
   - Review the extracted concepts, mathematical formulations, and student-friendly analogies.
   - Open **Agent Intent Routing & Classification Details** to demonstrate LLM intent routing and retrieval decisions.

4. **Presentation Studio & PowerPoint Export**:
   - Navigate to **Presentation Studio**.
   - Click **Generate Complete Presentation Deck**.
   - Review the 8-12 structured slides with bullet points, spoken Speaker Notes, and diagram recommendations.
   - Click **Download PowerPoint (.pptx)** and open the generated file in PowerPoint or a presentation viewer to show the custom 16:9 theme and embedded speaker notes!

5. **Viva Examination Studio & Interactive Simulator**:
   - Navigate to **Viva Examination Studio**.
   - Review the 6-tier examination syllabus with model defense answers.
   - Switch to **Interactive Oral Exam Simulator**.
   - Rehearse answering a question, submit the response, and review the AI Examiner's score (0-10), feedback on omitted technical concepts, and defense strategy.

6. **Grounded AI Research Chat & Document Explorer**:
   - In **AI Research Chat**, ask a targeted question and inspect the cited page numbers.
   - In **Document Explorer**, test the **Semantic Retrieval Sandbox** to demonstrate ChromaDB vector distance scoring.
#   P a p e r - t o - P r e s e n t a t i o n - V i v a - A g e n t  
 