"""Configuration constants and defaults for the Paper-to-Presentation & Viva Agent."""

DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_NUM_CTX = 4096
DEFAULT_TEMPERATURE = 0.2

# Text splitting settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ChromaDB collection
COLLECTION_NAME = "paper_viva_collection"

# Specialist agent keys and display titles
AGENT_DEFINITIONS = {
    "summary": {
        "title": "Summarizer Agent",
        "description": "Structured executive summary covering background, problem, methodology, findings, and conclusion.",
        "icon": "document"
    },
    "concepts": {
        "title": "Concept Extractor Agent",
        "description": "Extracts core technical concepts and explains them with intuition, math, and paper evidence.",
        "icon": "lightbulb"
    },
    "slides": {
        "title": "Slide Generator Agent",
        "description": "Creates an 8-10 slide conference/defense presentation with concise bullets and speaker notes.",
        "icon": "presentation"
    },
    "viva": {
        "title": "Viva Agent",
        "description": "Generates 10 examiner questions (basic to critical) with grounded model answers and defense tips.",
        "icon": "academic"
    },
    "explain": {
        "title": "Academic Explainer Agent",
        "description": "Answers specific questions about the paper using strictly retrieved evidence.",
        "icon": "search"
    }
}
