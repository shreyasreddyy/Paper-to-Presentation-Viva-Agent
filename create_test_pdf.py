"""Generate a clean, valid multi-page PDF research paper for testing PyPDFLoader."""

import pypdf

pages_text = [
    [
        "DeepResidualAttention: Enhanced Multi-Scale Representations in Deep Vision Models",
        "Authors: Dr. Elena Vance and Prof. Marcus Chen",
        "Abstract: Recent convolutional and vision transformer architectures demonstrate notable achievements",
        "in visual recognition. However, multi-scale feature degradation remains a critical challenge.",
        "We propose DeepResidualAttention (DRA), a novel hierarchical attention mechanism that dynamically",
        "balances spatial and channel-wise feature dependencies across distinct resolution scales.",
        "Our empirical evaluations on ImageNet-1K demonstrate that DRA achieves a top-1 accuracy improvement",
        "of 2.4 percent over ResNet-50 and 1.8 percent over Swin-T while reducing parameter count by 14.3 percent.",
        "1. Introduction and Research Problem:",
        "Deep residual networks rely on identity shortcut connections to alleviate gradient vanishing.",
        "Despite their success, deep features suffer from semantic attenuation as network depth scales.",
        "Existing spatial attention mechanisms often fail to capture long-range contextual relationships",
        "without introducing quadratic computational overhead. The primary research challenge is formulating",
        "a multi-scale attention weighting function that balances efficiency with representation power.",
    ],
    [
        "2. Proposed Methodology and Architecture:",
        "The DRA architecture introduces three core stages:",
        "1. Multi-Resolution Feature Pyramid Decomposition: The input tensor is decomposed into four stages.",
        "2. Cross-Scale Residual Gating: Each stage communicates through a bidirectional residual highway",
        "transferring high-frequency gradient signals to coarser semantic levels.",
        "3. Decoupled Spatial-Channel Attention Modules: We decouple spatial attention computation",
        "from channel pooling, reducing complexity from O(N^2) to O(N).",
        "We formulate the attention gating mechanism as:",
        "A(X) = sigma( W2 * GELU( W1 * Pool(X) ) )",
        "where X represents the input feature tensor, W1 and W2 denote down-projection and up-projection",
        "parameter matrices, and sigma denotes the sigmoid gating function. This formulation ensures that",
        "high-resolution spatial details are retained across all convolutional stages.",
    ],
    [
        "3. Experimental Setup and Quantitative Results:",
        "We evaluate DRA on ImageNet-1K (1.28 million training images, 50,000 validation images) and MS-COCO.",
        "Models are trained using AdamW optimizer with initial learning rate of 1e-4, cosine annealing",
        "over 300 epochs, batch size 256, and weight decay 0.05 on 8 NVIDIA A100 GPUs.",
        "Benchmark Comparison on ImageNet-1K:",
        "- ResNet-50 Baseline: 76.1 percent Top-1 Accuracy, 25.6M Parameters, 4.1 GFLOPs.",
        "- Swin-T Baseline: 81.3 percent Top-1 Accuracy, 28.3M Parameters, 4.5 GFLOPs.",
        "- DRA (Proposed): 83.1 percent Top-1 Accuracy, 24.2M Parameters, 3.9 GFLOPs.",
        "Ablation Studies:",
        "Removing cross-scale residual highway causes an immediate 1.7 percent drop in Top-1 accuracy,",
        "proving that inter-stage gradient communication is pivotal.",
        "4. Limitations and Threats to Validity:",
        "Although DRA reduces parameter count and FLOPs, memory footprint during training is approximately",
        "11 percent higher due to caching intermediate activation maps for cross-scale gating.",
        "5. Conclusion:",
        "DeepResidualAttention establishes that decoupled cross-scale attention gating achieves state-of-the-art",
        "accuracy with improved parameter efficiency. Future work will investigate edge deployment quantization.",
    ]
]


def build_pdf(pages_data, filename="sample_research_paper.pdf"):
    page_count = len(pages_data)
    # IDs: 1 = Catalog, 2 = Pages, then for each page: Page_i and Content_i, font is last
    cat_id = 1
    pages_id = 2
    font_id = 3 + 2 * page_count
    total_objs = font_id

    page_ids = [3 + 2 * i for i in range(page_count)]
    content_ids = [4 + 2 * i for i in range(page_count)]

    # Pre-generate content streams
    streams = []
    for lines in pages_data:
        cmds = ["BT", f"/F1 11 Tf", "50 720 Td"]
        for idx, line in enumerate(lines):
            safe_text = line.replace("\\", "").replace("(", "[").replace(")", "]")
            if idx > 0:
                cmds.append(f"0 -18 Td ({safe_text}) Tj")
            else:
                cmds.append(f"({safe_text}) Tj")
        cmds.append("ET")
        stream_bytes = "\n".join(cmds).encode("latin1")
        streams.append(stream_bytes)

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}

    def write_obj(oid, body):
        offsets[oid] = len(out)
        out.extend(f"{oid} 0 obj\n".encode("latin1"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    # 1. Catalog
    write_obj(cat_id, f"<</Type /Catalog /Pages {pages_id} 0 R>>".encode("latin1"))
    # 2. Pages
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    write_obj(pages_id, f"<</Type /Pages /Kids [{kids}] /Count {page_count}>>".encode("latin1"))

    # Pages and Contents
    for i in range(page_count):
        pid = page_ids[i]
        cid = content_ids[i]
        s_bytes = streams[i]
        write_obj(
            pid,
            f"<</Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Resources <</Font <</F1 {font_id} 0 R>>>> /Contents {cid} 0 R>>".encode("latin1"),
        )
        write_obj(
            cid,
            f"<</Length {len(s_bytes)}>>\nstream\n".encode("latin1") + s_bytes + b"\nendstream",
        )

    # Font
    write_obj(font_id, b"<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>")

    # Xref & Trailer
    startxref = len(out)
    out.extend(f"xref\n0 {total_objs + 1}\n0000000000 65535 f \n".encode("latin1"))
    for i in range(1, total_objs + 1):
        out.extend(f"{offsets[i]:010d} 00000 n \n".encode("latin1"))
    out.extend(f"trailer <</Size {total_objs + 1} /Root {cat_id} 0 R>>\nstartxref\n{startxref}\n%%EOF".encode("latin1"))

    with open(filename, "wb") as f:
        f.write(out)
    print(f"Generated {filename} ({len(out)} bytes).")


if __name__ == "__main__":
    build_pdf(pages_text)
