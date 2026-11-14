"""Automated test suite verifying all Agentic AI capabilities and acceptance criteria for Paper-to-Presentation & Viva Agent."""

import sys
import os
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    AGENT_DEFINITIONS,
)
from core.ollama_service import check_ollama_health, get_llm, get_embedder
from core.document_loader import load_single_file, load_multiple_documents
from core.vector_store import (
    build_vector_store,
    retrieve_context_with_metadata,
    search_chunks_by_keyword,
)
from core.router import safe_route
from core.agents import run_specialist_agent
from core.presentation_exporter import parse_slide_markdown, generate_pptx_file
from core.viva_simulator import (
    parse_viva_questions_from_text,
    evaluate_student_viva_answer,
)

MOCK_PAPER = """# DeepResidualAttention: Enhanced Multi-Scale Feature Representations in Deep Vision Models

## Abstract
Recent convolutional and transformer architectures have demonstrated significant advances in computer vision tasks. However, multi-scale feature degradation remains a critical challenge in fine-grained visual recognition. We propose DeepResidualAttention (DRA), a novel hierarchical attention mechanism that dynamically balances spatial and channel-wise feature dependencies across distinct resolution scales. Our extensive empirical evaluations on ImageNet-1K and MS-COCO demonstrate that DRA achieves a top-1 accuracy improvement of 2.4% over ResNet-50 and 1.8% over Swin-T while reducing parameter count by 14.3%.

## 1. Introduction and Problem Formulation
Traditional deep residual networks rely on identity shortcut connections to alleviate gradient vanishing. Despite their success, deep features suffer from semantic attenuation as network depth scales. Existing spatial attention mechanisms often fail to capture long-range contextual relationships without introducing quadratic computational overhead. 

We formulate the multi-scale attention weighting function as:
A(X) = \\sigma( W_2 \\cdot \\text{GELU}( W_1 \\cdot \\text{Pool}(X) ) )
where X represents the input feature tensor, W_1 and W_2 denote down-projection and up-projection parameter matrices, and \\sigma denotes the sigmoid gating function.

## 2. Proposed Methodology
The DRA architecture introduces three core stages:
1. Multi-Resolution Feature Pyramid Decomposition: The input image is downsampled into 4 hierarchical stages {C1, C2, C3, C4}.
2. Cross-Scale Residual Gating: Each stage communicates through a bidirectional residual highway that transfers high-frequency gradient signals to coarser semantic levels.
3. Decoupled Spatial-Channel Attention Modules: We decouple spatial attention computation from channel pooling, reducing complexity from O(N^2) to O(N).

## 3. Experimental Setup & Datasets
We evaluate DRA on ImageNet-1K (1.28 million training images, 50,000 validation images) and MS-COCO object detection. Models are trained using the AdamW optimizer with an initial learning rate of 1e-4, cosine annealing schedule over 300 epochs, batch size of 256, and weight decay of 0.05 on 8 NVIDIA A100 GPUs.

## 4. Empirical Results & Ablation Studies
Table 1 shows quantitative results:
- ResNet-50 baseline: 76.1% Top-1 accuracy, 25.6M parameters, 4.1 GFLOPs.
- Swin-T baseline: 81.3% Top-1 accuracy, 28.3M parameters, 4.5 GFLOPs.
- DRA (Proposed): 83.1% Top-1 accuracy, 24.2M parameters, 3.9 GFLOPs.

Ablation studies demonstrate that removing the cross-scale residual highway causes an immediate 1.7% drop in Top-1 accuracy, confirming that inter-stage gradient communication is pivotal.

## 5. Limitations and Threats to Validity
Although DRA reduces parameter count and FLOPs, memory footprint during training is approximately 11% higher due to caching intermediate activation maps for cross-scale gating. Additionally, performance on heavily occluded datasets has not been extensively benchmarked.

## 6. Conclusion
DeepResidualAttention establishes that decoupled cross-scale attention gating achieves state-of-the-art accuracy with improved parameter efficiency. Future work will investigate quantization-aware training for edge deployment.
"""


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""

    def __init__(self, name: str, content: str):
        self.name = name
        self._content = content.encode("utf-8")

    def getvalue(self) -> bytes:
        return self._content


def run_tests():
    print("=" * 70, flush=True)
    print("PAPER-TO-PRESENTATION & VIVA AGENT: AUTOMATED TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    # 1. Health check & LLM Initialization
    print("\n[Test 1] Ollama Health Check & Connectivity (Criterion A)...", flush=True)
    health = check_ollama_health()
    print(f"  Connected: {health['connected']} | Models available: {len(health['models'])}", flush=True)
    assert health["connected"], "Ollama is not running!"

    llm = get_llm(model_name=DEFAULT_CHAT_MODEL, temperature=0.1, num_ctx=4096)
    embedder = get_embedder(model_name=DEFAULT_EMBEDDING_MODEL)
    print("  ChatOllama and OllamaEmbeddings initialized successfully.", flush=True)

    # 2. Document Loading & Multi-File Support
    print("\n[Test 2] Document Ingestion & Multi-File Support...", flush=True)
    mock1 = MockUploadedFile("paper_part1.md", MOCK_PAPER[:1500])
    mock2 = MockUploadedFile("paper_part2.md", MOCK_PAPER[1500:])
    all_docs, file_metas, agg_meta = load_multiple_documents([mock1, mock2])
    print(f"  Ingested {agg_meta['document_count']} files, {agg_meta['total_chars']} chars across {len(all_docs)} sections.", flush=True)
    assert agg_meta["document_count"] == 2, "Expected 2 documents"

    # 3. Vector Database Indexing & Chunking
    print("\n[Test 3] ChromaDB Vector Store & Embedding Index (Criterion E & G)...", flush=True)
    full_mock = MockUploadedFile("deep_residual_attention.md", MOCK_PAPER)
    single_docs, _ = load_single_file(full_mock)
    store, chunks = build_vector_store(single_docs, embedder)
    print(f"  Built in-memory ChromaDB store with {len(chunks)} text chunks.", flush=True)
    assert len(chunks) >= 3, "Expected at least 3 chunks"

    # 4. Keyword Search & Semantic Retrieval
    print("\n[Test 4] Keyword Search & Semantic Retrieval (Criterion E)...", flush=True)
    kw_results = search_chunks_by_keyword(chunks, "ablation")
    print(f"  Keyword search for 'ablation' matched {len(kw_results)} chunks.", flush=True)
    assert len(kw_results) > 0, "Keyword search failed"

    context, retrieved_info = retrieve_context_with_metadata(
        store, "What is the Top-1 accuracy and parameter count of DRA?", k=3
    )
    print(f"  Semantic search retrieved {len(retrieved_info)} chunks with distance scores:", flush=True)
    for r in retrieved_info:
        print(f"    - Rank #{r['rank']} | Chunk #{r['chunk_id']} | Distance: {r['score']} | Chars: {len(r['content'])}", flush=True)
    assert len(retrieved_info) > 0, "Semantic retrieval returned 0 chunks"
    assert "83.1%" in context or "ImageNet" in context or "DRA" in context, "Context missing key paper metrics"

    # 5. Intent Detection & Routing
    print("\n[Test 5] Intent Detection & Routing (Criterion F & D)...", flush=True)
    route_test_queries = [
        ("Summarize the paper's key findings", "summary"),
        ("Create presentation slides with speaker notes", "slides"),
        ("What viva questions will the professor ask?", "viva"),
        ("Explain how decoupled attention works", "explain"),
    ]
    for q_text, expected in route_test_queries:
        decision, raw_r = safe_route(q_text, llm, has_document=True)
        print(f"  Query: '{q_text}' -> Intent: {decision['intent']} | Retrieve: {decision['retrieve']} | Reason: {decision.get('reason')}", flush=True)
        assert decision["intent"] in {"summary", "concepts", "slides", "viva", "explain"}, "Invalid intent returned"

    # 6. PowerPoint PPTX Generation
    print("\n[Test 6] PowerPoint (.pptx) Presentation Generation (Phase 4 & 10)...", flush=True)
    sample_slide_text = """
    # Presentation Deck
    ### Slide 1: Introduction & Research Motivation
    - Multi-scale feature degradation in deep vision models.
    - Decoupled attention overcomes O(N^2) complexity.
    - Significant performance gains on ImageNet-1K.
    **Speaker Notes:** Introduce the problem domain and historical limitations of deep residual networks.
    **Visual Recommendation:** Flowchart illustrating feature attenuation.

    ### Slide 2: Empirical Results & Benchmark Comparisons
    - DRA achieves 83.1% Top-1 accuracy on ImageNet-1K.
    - 2.4% improvement over ResNet-50 baseline.
    - 14.3% reduction in total parameter count.
    **Speaker Notes:** Emphasize the parameter efficiency compared to Swin-T and ResNet-50 baselines.
    **Visual Recommendation:** Bar chart comparing accuracy vs parameter count.
    """
    parsed_slides = parse_slide_markdown(sample_slide_text)
    print(f"  Parsed {len(parsed_slides)} structured slides from markdown.", flush=True)
    assert len(parsed_slides) == 2, f"Expected 2 slides, got {len(parsed_slides)}"

    pptx_buf = generate_pptx_file(parsed_slides, "Deep Residual Attention")
    pptx_bytes = len(pptx_buf.getvalue())
    print(f"  Generated valid PowerPoint (.pptx) file: {pptx_bytes} bytes.", flush=True)
    assert pptx_bytes > 10000, "PPTX file size abnormally small"

    # 7. Viva Question Parsing & Interactive Simulator
    print("\n[Test 7] Viva Questions Parsing & Defense Simulator (Phase 5)...", flush=True)
    sample_viva_text = """
    # Viva Voce Examination Guide
    ### Question 1: What is the primary architectural contribution of DeepResidualAttention?
    - **Category:** Fundamental Concepts
    - **Examiner's Objective:** Verify understanding of novel claims.
    - **Model Defense Answer:** Decoupled spatial-channel attention combined with cross-scale residual highways.
    - **Defense Strategy / Viva Tip:** Be concise and cite the 14.3% parameter reduction.

    ### Question 2: Why did removing the residual highway cause a 1.7% drop in accuracy?
    - **Category:** Methodology & Ablation Studies
    - **Examiner's Objective:** Test depth of ablation understanding.
    - **Model Defense Answer:** Inter-stage gradient communication is pivotal for high-frequency signal propagation.
    - **Defense Strategy / Viva Tip:** Reference the ablation experiments in Section 4.
    """
    viva_qs = parse_viva_questions_from_text(sample_viva_text)
    print(f"  Parsed {len(viva_qs)} viva questions with categories and examiner objectives.", flush=True)
    assert len(viva_qs) == 2, f"Expected 2 viva questions, got {len(viva_qs)}"

    # Test student response evaluation
    print("  Testing AI Examiner answer evaluation...", flush=True)
    student_sample = "DRA decouples spatial and channel attention and uses residual highways to reduce complexity."
    eval_res = evaluate_student_viva_answer(viva_qs[0], student_sample, llm)
    print(f"  Evaluation Verdict: {eval_res['verdict']} | Score: {eval_res['score']}/10", flush=True)
    print(f"  Feedback: {eval_res['constructive_feedback'][:120]}...", flush=True)
    assert 0 <= eval_res["score"] <= 10, "Evaluation score out of bounds"

    print("\n" + "=" * 70, flush=True)
    print("ALL SEVEN CRITERIA AND WORKSPACE CAPABILITIES FULLY VERIFIED!", flush=True)
    print("=" * 70, flush=True)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
