
# Paper-to-Presentation & Viva Agent

A local-first, privacy-preserving Agentic AI research assistant and thesis defense workspace designed for academic research, presentation generation, technical explanations, and viva preparation.

Built using **Python 3.11**, **Streamlit**, **LangChain (LCEL)**, **Ollama (Llama 3.2)**, **ChromaDB**, **PyPDF**, and **python-pptx**.

---

## 1. Project Overview

Paper-to-Presentation & Viva Agent is an AI-powered academic assistant that helps students understand research papers and prepare for academic presentations and viva examinations.

The application uses locally hosted language models and Retrieval-Augmented Generation (RAG) to process research documents and generate relevant academic content.

### Key Features

- Research paper summarization.
- Technical concept extraction and explanations.
- AI-powered presentation generation.
- Downloadable PowerPoint presentations.
- Automated viva question generation.
- Interactive viva examination simulator.
- Context-aware academic question answering.
- Semantic document search.
- Local AI processing using Ollama.

---

## 2. System Architecture

The application implements an Agentic Retrieval-Augmented Generation (RAG) pipeline using local infrastructure.

```text
              USER UPLOAD
             (PDF / TXT / MD)
                    |
                    v
             DOCUMENT INGESTION
                    |
                    v
              TEXT EXTRACTION
                    |
                    v
              SEMANTIC CHUNKING
          (RecursiveCharacterTextSplitter)
                    |
                    v
              VECTOR EMBEDDINGS
             (OllamaEmbeddings)
                    |
                    v
             CHROMADB VECTOR STORE
                    |
                    v
              INTENT ROUTER
                    |
          +---------+---------+
          |                   |
          v                   v
    SPECIALIST QUERY     ACADEMIC CHAT
          |                   |
          v                   v
    RAG RETRIEVAL       SEMANTIC SEARCH
          |                   |
          v                   v
    SPECIALIST AGENTS   GROUNDED ANSWERS
          |
    +-----+------+-------+-------+
    |     |      |       |       |
    v     v      v       v       v
 SUMMARY CONCEPT SLIDES  VIVA  EXPLAINER
    |     |      |       |       |
    +-----+------+-------+-------+
                    |
                    v
              FINAL OUTPUT
         SUMMARY / PPTX / VIVA / Q&A
```

---

## 3. Seven Mandatory Acceptance Criteria

| # | Acceptance Criterion | Technical Implementation |
|---|---|---|
| A | LLM Integration | Ollama and ChatOllama |
| B | Interactive GUI | Streamlit interface |
| C | Prompt Engineering | Role-based prompts and structured outputs |
| D | LangChain Core & LCEL | LangChain prompt pipelines |
| E | Agentic RAG | ChromaDB semantic retrieval |
| F | Intent Detection & Routing | Intent classification and routing |
| G | Vector Database | ChromaDB with Ollama embeddings |

### A. LLM Integration

- Integrates locally hosted Llama 3.2.
- Uses LangChain's ChatOllama integration.
- Supports local CPU/GPU inference depending on hardware.
- Monitors Ollama connectivity.

### B. Interactive GUI

The Streamlit application provides an interactive academic workspace with dedicated sections for:

- Dashboard
- Research Studio
- Presentation Studio
- Viva Examination Studio
- AI Research Chat
- Document Explorer
- Architecture Diagnostics

### C. Prompt Engineering

- Separate system and user prompts.
- Role-based AI agent instructions.
- Structured output generation.
- Context-aware responses.
- Grounding instructions to reduce hallucinations.

### D. LangChain Core & LCEL

Uses LangChain Expression Language to construct AI processing pipelines.

Example:

```python
chain = prompt | llm | StrOutputParser()
```

### E. Agentic RAG

- Extracts content from uploaded research documents.
- Splits documents into smaller chunks.
- Generates vector embeddings.
- Retrieves relevant passages using semantic similarity.
- Provides context-grounded academic responses.

### F. Intent Detection & Routing

The intent router identifies the user's requested task and directs it to the appropriate specialist agent.

Supported task categories include:

- Summary
- Concepts
- Slides
- Viva
- Explain

### G. Vector Database

- Uses ChromaDB for vector storage.
- Uses Ollama embeddings.
- Supports semantic similarity search.
- Maintains document context for retrieval.

---

## 4. Specialist AI Agent Team

### 1. Research Summarizer Agent

Generates structured research paper summaries covering:

- Background and research motivation.
- Core problem formulation.
- Research objectives.
- Novel contributions.
- Methodology.
- Experimental evaluation.
- Results and key findings.
- Limitations.
- Conclusion and future scope.

### 2. Concept Extractor Agent

Identifies important technical concepts and explains them in student-friendly language.

Provides:

- Academic definitions.
- Simple explanations.
- Real-world analogies.
- Technical formulations.
- Applications in the research paper.
- Architectural significance.

### 3. Presentation Generator Agent

Automatically generates academic presentation content.

Features include:

- Structured slide progression.
- Research problem and objectives.
- Related work.
- Proposed methodology.
- System architecture.
- Experimental results.
- Limitations and conclusion.
- Presenter speaker notes.
- Visual and diagram recommendations.
- Downloadable PowerPoint (.pptx) output.

### 4. Viva Preparation Agent

Prepares students for academic viva examinations through multiple difficulty levels.

#### Six Viva Preparation Tiers

- **Tier 1:** Basic and foundational concepts.
- **Tier 2:** Problem domain and literature justification.
- **Tier 3:** Methodology, algorithms and architecture.
- **Tier 4:** Experimental setup, datasets and metrics.
- **Tier 5:** Results, validation and evaluation.
- **Tier 6:** Critical defence, limitations and future scope.

#### Interactive Viva Simulator

- Generates viva questions.
- Accepts student answers.
- Evaluates responses using AI.
- Provides feedback.
- Identifies missing technical concepts.
- Suggests improved answers.
- Provides an AI examiner score.

### 5. Academic Explainer Agent

Provides contextual explanations and answers based on the uploaded research paper.

Features:

- Academic question answering.
- Context-aware explanations.
- Relevant passage retrieval.
- Source-based answers.
- Page and chunk references where available.

---

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Streamlit | Interactive user interface |
| LangChain | AI application framework |
| LCEL | AI processing pipelines |
| Ollama | Local LLM execution |
| Llama 3.2 | Language model |
| Nomic Embed Text | Text embeddings |
| ChromaDB | Vector database |
| PyPDF | PDF text extraction |
| RapidOCR | Scanned document OCR |
| python-pptx | PowerPoint generation |

---

## 6. Project Structure

```text
Paper-to-Presentation-Viva-Agent/
│
├── app.py
│
├── core/
│   ├── __init__.py
│   ├── agents.py
│   ├── config.py
│   ├── document_loader.py
│   ├── ollama_service.py
│   ├── presentation_exporter.py
│   ├── router.py
│   ├── vector_store.py
│   └── viva_simulator.py
│
├── requirements.txt
│
├── sample_research_paper.pdf
│
├── create_test_pdf.py
├── test_ocr_pipeline.py
├── test_system.py
├── test_workflow.py
├── verify_submission_ready.py
│
├── .gitignore
├── README.md
├── setup_windows.bat
└── run_windows.bat
```

---

## 7. Installation and Setup

### Prerequisites

- Python 3.11 (64-bit recommended).
- Ollama installed.
- Windows 10/11, macOS or Linux.
- Sufficient RAM for local AI models.

### Step 1: Install Ollama

Download Ollama:

https://ollama.com/

### Step 2: Download Required Models

Open Command Prompt or PowerShell:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify the installed models:

```bash
ollama list
```

### Step 3: Create a Python Virtual Environment

Navigate to the project directory:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

The application will open at:

http://localhost:8501

---

## 8. How to Use the Application

1. Start the Ollama service.
2. Launch the Streamlit application.
3. Upload a research paper in PDF, TXT or Markdown format.
4. Click the document processing option.
5. Wait for document extraction and vector indexing.
6. Select the required specialist AI agent.
7. Enter your task or question.
8. Generate the AI response.
9. Review the generated academic content.
10. Download the presentation when using the presentation generator.

---

## 9. College Project Demonstration

The following demonstration workflow can be used during a project review.

### Demonstration 1: Dashboard

- Show the application dashboard.
- Demonstrate Ollama connectivity.
- Explain the local AI architecture.
- Introduce the available specialist agents.

### Demonstration 2: Research Paper Processing

- Upload a research paper.
- Start document processing.
- Demonstrate text extraction.
- Explain semantic chunking.
- Show vector indexing.

### Demonstration 3: Specialist AI Agents

- Generate a research summary.
- Extract technical concepts.
- Explain the research methodology.
- Demonstrate AI-generated academic responses.

### Demonstration 4: Presentation Generation

- Open Presentation Studio.
- Generate presentation content.
- Review slide titles and bullet points.
- Review speaker notes.
- Download the PowerPoint presentation.

### Demonstration 5: Viva Preparation

- Open Viva Examination Studio.
- Generate viva questions.
- Review questions across different difficulty levels.
- Answer questions using the interactive simulator.
- Review AI feedback and evaluation.

### Demonstration 6: Academic Research Chat

- Ask questions about the uploaded research paper.
- Demonstrate contextual retrieval.
- Review relevant source passages.
- Explain how RAG improves response relevance.

---

## 10. Testing and Validation

Run Python syntax checks:

```bash
python -m compileall -q app.py core
```

Run the system integration test:

```bash
python test_system.py
```

Run the OCR pipeline test:

```bash
python test_ocr_pipeline.py
```

Run the complete workflow test:

```bash
python test_workflow.py
```

**Note:** Integration and workflow tests require Ollama to be running with the required models installed. A successful syntax check alone does not guarantee that the complete application works.

---

## 11. Privacy and Security

- Uses locally hosted Ollama models.
- Does not require paid LLM API keys for its default local workflow.
- Processes research documents locally.
- Uses an in-memory ChromaDB vector store.
- Keeps API credentials out of source code.
- Uses document-based retrieval for contextual responses.

**Note:** Local processing depends on the configured application and its dependencies. Verify that no optional external services are enabled before describing the application as entirely offline.

---

## 12. Limitations

- AI-generated content may contain inaccurate information.
- Response quality depends on the selected language model.
- Large documents may require additional processing time.
- Scanned PDFs depend on OCR quality.
- Local inference requires sufficient hardware resources.
- The current Streamlit upload workflow processes one document at a time.
- Generated presentations should be reviewed before academic use.

---

## 13. Future Enhancements

- Multi-document research analysis.
- Advanced presentation themes.
- Automatic architecture diagram generation.
- More advanced viva evaluation.
- Voice-based viva interaction.
- Multilingual academic explanations.
- Export to additional document formats.
- Persistent research history.
- Improved citation and reference management.

---

## 14. Conclusion

Paper-to-Presentation & Viva Agent demonstrates how Agentic AI, Retrieval-Augmented Generation, local language models and vector databases can be combined to simplify academic research and presentation preparation.

By integrating multiple specialist AI agents into a single application, the system helps students understand complex research papers, generate presentation content and prepare for technical viva examinations.

The project demonstrates practical applications of Large Language Models, prompt engineering, LangChain, semantic search, vector databases and interactive AI systems.

---

## 15. Credits and Licensing

This project may incorporate code or ideas from other contributors.

All original authors and contributors should receive appropriate attribution. Obtain permission to reuse or redistribute third-party code and retain applicable license notices.

Add a software license only after confirming that you have the right to distribute the project under that license.
