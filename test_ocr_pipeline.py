"""End-to-end verification script for Digital and Scanned PDF Ingestion with Local OCR."""

import io
import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.document_loader import load_uploaded_document
from core.vector_store import build_vector_store, retrieve_context_with_metadata
from core.ollama_service import get_embedder, get_llm
from core.agents import run_specialist_agent


class DummyUploadedFile:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def create_scanned_pdf(output_path: str):
    """Generate a multi-page scanned/image-only PDF with zero digital text layer."""
    pages_content = [
        [
            "Research Paper: Neural Architecture for Edge Intelligence",
            "Author: Dr. Alex Turner, Quantum AI Labs",
            "Abstract: This research investigates low-power edge computing for deep neural networks.",
            "Edge devices frequently face computational constraints and thermal throttling limits.",
            "Our proposed system introduces quantized depthwise separable convolutions to save energy.",
        ],
        [
            "Methodology & Experimental Setup",
            "We deployed the model on ARM Cortex-M55 microcontrollers running at 160MHz.",
            "Weight quantization was performed down to INT8 precision with minimal accuracy degradation.",
            "Inference latency was benchmarked across 10,000 continuous test cycles.",
            "Memory footprint was constrained strictly under 256 Kilobytes of SRAM.",
        ],
        [
            "Results and Conclusion",
            "The quantized architecture achieved 94.2% top-1 accuracy on CIFAR-10.",
            "Energy consumption dropped by 68% compared to uncompressed baseline models.",
            "In conclusion, edge intelligence is achievable without dedicated cloud connectivity.",
        ],
    ]

    images = []
    for page_lines in pages_content:
        # Create an 1200x800 white image representing a scanned printed document page
        img = Image.new("RGB", (1200, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = 60
        for line in page_lines:
            # Draw black text onto the canvas (rasterized image only, no PDF text stream)
            draw.text((60, y), line, fill=(0, 0, 0))
            y += 60

        images.append(img)

    # Save as multi-page image PDF
    images[0].save(output_path, save_all=True, append_images=images[1:], resolution=150.0)
    print(f"Created synthetic scanned PDF with {len(images)} pages: '{output_path}'")


def create_blank_scanned_pdf(output_path: str):
    """Generate a completely blank scanned PDF to test OCR error handling."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    img.save(output_path)
    print(f"Created blank scanned PDF: '{output_path}'")


def main():
    print("=" * 70)
    print("TESTING ROBUST PDF INGESTION (DIGITAL + LOCAL OCR)")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST 1: Digital PDF Ingestion (sample_research_paper.pdf)
    # -------------------------------------------------------------
    print("\n[TEST 1] Digital PDF Ingestion (Existing PyPDF)...")
    with open("sample_research_paper.pdf", "rb") as f:
        digital_bytes = f.read()

    digital_file = DummyUploadedFile("sample_research_paper.pdf", digital_bytes)
    dig_docs, dig_meta = load_uploaded_document(digital_file)
    print(f"  Method detected: {dig_meta['extraction_method']}")
    print(f"  Pages: {dig_meta['pages']}, Chars: {dig_meta['total_chars']}")
    assert "Digital" in dig_meta["extraction_method"], "Expected digital extraction method"
    assert dig_meta["total_chars"] > 500, "Expected significant digital characters"
    assert len(dig_docs) > 0, "Expected document objects"
    print("  --> [PASS] Digital PDF Ingestion succeeded.\n")

    # -------------------------------------------------------------
    # TEST 2: Scanned PDF Ingestion with Local OCR
    # -------------------------------------------------------------
    print("[TEST 2] Scanned PDF Ingestion (RapidOCR + pypdfium2)...")
    scanned_path = "scanned_research_paper.pdf"
    create_scanned_pdf(scanned_path)

    with open(scanned_path, "rb") as f:
        scanned_bytes = f.read()

    scanned_file = DummyUploadedFile("scanned_research_paper.pdf", scanned_bytes)
    ocr_docs, ocr_meta = load_uploaded_document(scanned_file)
    print(f"  Method detected: {ocr_meta['extraction_method']}")
    print(f"  Pages: {ocr_meta['pages']}, Chars: {ocr_meta['total_chars']}")
    print(f"  Extracted Document Pages: {len(ocr_docs)}")
    for d in ocr_docs:
        print(f"    - Page {d.metadata['page']}: {len(d.page_content)} chars | Sample: '{d.page_content[:60]}...'")

    assert "OCR" in ocr_meta["extraction_method"], "Expected OCR extraction method"
    assert ocr_meta["total_chars"] > 100, "Expected OCR to extract text"
    assert len(ocr_docs) == 3, f"Expected 3 pages from scanned PDF, got {len(ocr_docs)}"
    print("  --> [PASS] Scanned PDF Local OCR extraction succeeded.\n")

    # -------------------------------------------------------------
    # TEST 3: ChromaDB Vector Indexing & RAG on OCR Content
    # -------------------------------------------------------------
    print("[TEST 3] Building ChromaDB Vector Index from OCR Documents...")
    embedder = get_embedder()
    store, chunks = build_vector_store(ocr_docs, embedder)
    print(f"  Indexed chunks in ChromaDB: {len(chunks)}")
    assert len(chunks) > 0, "Expected indexed chunks"

    # Query ChromaDB
    context_str, retrieved = retrieve_context_with_metadata(store, "What accuracy was achieved on CIFAR-10?", k=2)
    print(f"  Retrieved {len(retrieved)} relevant chunks:")
    for r in retrieved:
        print(f"    - Page {r['page']} (score: {r['score']}): {r['content'][:80]}...")
    assert len(retrieved) > 0, "Expected retrieval results"
    assert any("94.2%" in r["content"] or "accuracy" in r["content"].lower() for r in retrieved), "Expected retrieval of CIFAR-10 result"
    print("  --> [PASS] Vector indexing and context retrieval on OCR content succeeded.\n")

    # -------------------------------------------------------------
    # TEST 4: Agent Execution on Scanned OCR Document
    # -------------------------------------------------------------
    print("[TEST 4] Running Academic Explainer Agent on OCR Context...")
    llm = get_llm()
    answer = run_specialist_agent("explain", store, "What is the memory footprint and accuracy of the model?", llm)
    print("  Agent Response Preview:")
    print("  " + answer[:300].replace("\n", "\n  ") + "...\n")
    assert len(answer) > 50, "Expected non-empty agent response"
    print("  --> [PASS] Agent successfully answered questions from scanned OCR context.\n")

    # -------------------------------------------------------------
    # TEST 5: Markdown and Text Document Ingestion
    # -------------------------------------------------------------
    print("[TEST 5] Verifying Markdown/TXT document ingestion...")
    md_content = b"# Paper Title\n\n## Abstract\nThis is a sample markdown paper for testing."
    md_file = DummyUploadedFile("notes.md", md_content)
    md_docs, md_meta = load_uploaded_document(md_file)
    print(f"  Method detected: {md_meta['extraction_method']}")
    assert md_meta["extraction_method"] == "Direct Text"
    assert len(md_docs) == 1
    print("  --> [PASS] Markdown ingestion verified.\n")

    # -------------------------------------------------------------
    # TEST 6: Blank Scanned PDF Error Handling
    # -------------------------------------------------------------
    print("[TEST 6] Testing blank scanned PDF error handling...")
    blank_path = "blank_scanned.pdf"
    create_blank_scanned_pdf(blank_path)
    with open(blank_path, "rb") as f:
        blank_bytes = f.read()

    blank_file = DummyUploadedFile("blank_scanned.pdf", blank_bytes)
    caught_error = False
    try:
        load_uploaded_document(blank_file)
    except ValueError as e:
        caught_error = True
        print(f"  Successfully caught expected error: {e}")

    assert caught_error, "Expected ValueError on blank scanned PDF"
    print("  --> [PASS] Blank scanned PDF handled with meaningful error.\n")

    print("=" * 70)
    print("ALL TESTS PASSED: DIGITAL & SCANNED LOCAL OCR PIPELINE FULLY FUNCTIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
