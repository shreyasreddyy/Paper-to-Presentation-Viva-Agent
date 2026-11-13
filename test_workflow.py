"""End-to-End Workflow Verification Script for Paper-to-Presentation & Viva Agent.
Tests document upload, chunking, ChromaDB indexing, intent routing, all 4 specialist agents, and PPTX generation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ollama_service import check_ollama_health, get_llm, get_embedder
from core.document_loader import load_uploaded_document
from core.vector_store import build_vector_store, retrieve_context_with_metadata
from core.router import safe_route
from core.agents import run_specialist_agent
from core.presentation_exporter import parse_slide_markdown, generate_pptx_file


class MockUploadedPDF:
    def __init__(self, filepath):
        self.name = Path(filepath).name
        with open(filepath, "rb") as f:
            self._bytes = f.read()

    def getvalue(self):
        return self._bytes


def main():
    print("=" * 75, flush=True)
    print("STARTING REAL END-TO-END WORKFLOW VERIFICATION", flush=True)
    print("=" * 75, flush=True)

    # 1. Ollama Health
    print("\n[Step 1] Checking Ollama Infrastructure...", flush=True)
    health = check_ollama_health()
    print(f"  Ollama Status: {'ONLINE' if health['connected'] else 'OFFLINE'}", flush=True)
    assert health["connected"], "Ollama is not running!"

    llm = get_llm(model_name="llama3.2", temperature=0.1, num_ctx=4096)
    embedder = get_embedder(model_name="nomic-embed-text")
    print("  Models configured: llama3.2 (Chat) & nomic-embed-text (Embeddings).", flush=True)

    # 2. Document Upload & Extraction
    print("\n[Step 2] Testing PDF Upload & Text Extraction...", flush=True)
    pdf_file = "sample_research_paper.pdf"
    mock_upload = MockUploadedPDF(pdf_file)
    docs, meta = load_uploaded_document(mock_upload)

    print(f"  Extracted: '{meta['filename']}'", flush=True)
    print(f"  Total Pages: {meta['pages']} | Readable Pages: {meta['readable_pages']}", flush=True)
    print(f"  Total Characters: {meta['total_chars']} | Size: {meta['file_size_kb']} KB", flush=True)
    print(f"  Document SHA-256 Hash: {meta['file_hash'][:16]}...", flush=True)
    assert meta["pages"] == 3, f"Expected 3 pages, got {meta['pages']}"
    assert meta["total_chars"] > 2000, "Text extraction returned too few characters"

    # 3. Text Chunking & ChromaDB Indexing
    print("\n[Step 3] Splitting Text and Indexing into ChromaDB...", flush=True)
    t0 = time.time()
    store, chunks = build_vector_store(docs, embedder)
    indexing_time = round(time.time() - t0, 2)
    print(f"  Successfully built ChromaDB index with {len(chunks)} chunks in {indexing_time}s.", flush=True)
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    # 4. Agentic RAG Retrieval
    print("\n[Step 4] Testing Agentic Retrieval with Metadata...", flush=True)
    query = "What is the Top-1 accuracy improvement of DeepResidualAttention over ResNet-50?"
    context, retrieved_chunks = retrieve_context_with_metadata(store, query, k=3)
    print(f"  Retrieved {len(retrieved_chunks)} relevant chunks from ChromaDB:", flush=True)
    for c in retrieved_chunks:
        print(f"    - Rank #{c['rank']} | Page {c['page']} | Distance: {c['score']} | {c['content'][:80]}...", flush=True)
    assert len(retrieved_chunks) > 0, "No chunks retrieved"
    assert "2.4" in context or "83.1" in context or "ResNet-50" in context, "Retrieved context does not contain relevant paper facts"

    # 5. Intent Routing Verification
    print("\n[Step 5] Testing Intent Detection & Retrieval Enforcement...", flush=True)
    route_tests = [
        ("Summarize the paper's key methodology and findings", "summary", True),
        ("Extract and explain the core concepts in simple terms", "concepts", True),
        ("Create 8 to 10 presentation slides with speaker notes", "slides", True),
        ("Generate 10 viva defense questions with model answers", "viva", True),
        ("What is machine learning in general?", "explain", False),
    ]
    for q_text, expected_intent, expected_retrieve in route_tests:
        decision, _ = safe_route(q_text, llm, has_document=True)
        print(f"  Query: '{q_text[:40]}...' -> Intent: {decision['intent']} | Retrieve: {decision['retrieve']}", flush=True)
        assert decision["intent"] == expected_intent, f"Expected {expected_intent}, got {decision['intent']}"
        assert decision["retrieve"] == expected_retrieve, f"Expected retrieve={expected_retrieve}, got {decision['retrieve']}"

    # 6. Specialist Agent A: Summarizer
    print("\n[Step 6] Testing Agent 1: Summarizer Agent...", flush=True)
    sum_query = "Provide a structured summary containing abstract, problem, objectives, methodology, results, and conclusion."
    sum_context, _ = retrieve_context_with_metadata(store, sum_query, k=4)
    summary_out = run_specialist_agent("summary", sum_context, sum_query, llm)
    print(f"  Generated Summary ({len(summary_out)} chars).", flush=True)
    assert "Abstract" in summary_out or "Problem" in summary_out or "Methodology" in summary_out, "Summary missing key headings"
    print("  Preview:\n" + "\n".join(summary_out.split("\n")[:8]) + "\n...", flush=True)

    # 7. Specialist Agent B: Concept Extractor
    print("\n[Step 7] Testing Agent 2: Concept Extractor Agent...", flush=True)
    con_query = "Extract key technical concepts and explain each in student-friendly terms with concrete examples."
    con_context, _ = retrieve_context_with_metadata(store, con_query, k=4)
    concepts_out = run_specialist_agent("concepts", con_context, con_query, llm)
    print(f"  Generated Concepts ({len(concepts_out)} chars).", flush=True)
    assert "Concept" in concepts_out or "Definition" in concepts_out, "Concepts output missing expected format"

    # 8. Specialist Agent C: Presentation Generator & PPTX Export
    print("\n[Step 8] Testing Agent 3: Presentation Generator & PowerPoint Exporter...", flush=True)
    slide_query = "Generate an 8 to 10 slide presentation deck with bullet points, spoken speaker notes, and diagram suggestions."
    slide_context, _ = retrieve_context_with_metadata(store, slide_query, k=5)
    slides_out = run_specialist_agent("slides", slide_context, slide_query, llm)
    print(f"  Generated Slides Markdown ({len(slides_out)} chars).", flush=True)
    assert "Slide" in slides_out, "Slides output missing Slide headers"

    parsed_slides = parse_slide_markdown(slides_out)
    print(f"  Parsed {len(parsed_slides)} structured slides.", flush=True)
    assert len(parsed_slides) >= 4, f"Expected at least 4 parsed slides, got {len(parsed_slides)}"

    pptx_buf = generate_pptx_file(parsed_slides, paper_title="Deep Residual Attention")
    pptx_bytes = len(pptx_buf.getvalue())
    print(f"  Created PowerPoint (.pptx) file: {pptx_bytes} bytes.", flush=True)
    assert pptx_bytes > 20000, "PPTX file size abnormally small"

    # 9. Specialist Agent D: Viva Generator
    print("\n[Step 9] Testing Agent 4: Viva Generator Agent...", flush=True)
    viva_query = "Generate 10 relevant viva questions and answers covering basic, intermediate, and advanced levels."
    viva_context, _ = retrieve_context_with_metadata(store, viva_query, k=5)
    viva_out = run_specialist_agent("viva", viva_context, viva_query, llm)
    print(f"  Generated Viva Guide ({len(viva_out)} chars).", flush=True)
    assert "Question" in viva_out, "Viva output missing Question headers"

    # 10. Verification of Grounding in Uploaded Paper
    print("\n[Step 10] Verifying Paper Grounding across outputs...", flush=True)
    combined_outputs = summary_out + " " + concepts_out + " " + slides_out + " " + viva_out
    has_paper_facts = (
        "DeepResidualAttention" in combined_outputs
        or "DRA" in combined_outputs
        or "ResNet-50" in combined_outputs
        or "ImageNet" in combined_outputs
        or "2.4" in combined_outputs
    )
    print(f"  Grounded in actual paper content: {has_paper_facts}", flush=True)
    assert has_paper_facts, "Outputs were not grounded in the uploaded paper!"

    print("\n" + "=" * 75, flush=True)
    print("ALL WORKFLOW STEPS AND AGENTS TESTED AND FULLY FUNCTIONAL!", flush=True)
    print("=" * 75, flush=True)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
