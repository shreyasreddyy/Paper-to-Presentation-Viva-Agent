"""Core modules for Paper-to-Presentation & Viva Agent."""

from core.document_loader import load_uploaded_document, load_multiple_documents, load_single_file
from core.vector_store import build_vector_store, retrieve_context_with_metadata, search_chunks_by_keyword
from core.router import safe_route
from core.agents import run_specialist_agent, make_agent_chain
from core.presentation_exporter import parse_slide_markdown, generate_pptx_file
from core.viva_simulator import parse_viva_questions_from_text, evaluate_student_viva_answer
from core.ollama_service import check_ollama_health, get_llm, get_embedder

__all__ = [
    "load_uploaded_document",
    "load_multiple_documents",
    "load_single_file",
    "build_vector_store",
    "retrieve_context_with_metadata",
    "search_chunks_by_keyword",
    "safe_route",
    "run_specialist_agent",
    "make_agent_chain",
    "parse_slide_markdown",
    "generate_pptx_file",
    "parse_viva_questions_from_text",
    "evaluate_student_viva_answer",
    "check_ollama_health",
    "get_llm",
    "get_embedder",
]
