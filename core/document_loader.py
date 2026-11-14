"""Document loading, validation, digital text extraction, and local OCR fallback.

Features:
- Primary extraction: PyPDF for high-speed digital text extraction.
- Automatic detection of scanned / image-only PDFs with zero or insufficient text.
- Offline local OCR fallback: pypdfium2 (in-memory rendering) + RapidOCR (ONNXRuntime).
- Page-by-page extraction with preserved page numbers.
- Enriched LangChain Document objects ready for ChromaDB vector indexing.
- Comprehensive support for PDF, TXT, and Markdown files.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Module-level lazy singleton for RapidOCR to prevent reloading ONNX models on every invocation
_OCR_ENGINE = None


def _get_ocr_engine():
    """Retrieve or initialize the singleton RapidOCR engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except ImportError as exc:
            raise RuntimeError(
                "Local OCR dependencies are missing. Please install 'rapidocr-onnxruntime' "
                "and 'pypdfium2' via pip to enable scanned document processing."
            ) from exc
    return _OCR_ENGINE


def _extract_digital_pdf(temp_path: str) -> Tuple[List[Tuple[int, str]], int]:
    """Extract digital text from PDF pages using PyPDFLoader.

    Returns:
        List of (1-based page number, text content) and total character count.
    """
    loader = PyPDFLoader(temp_path)
    raw_docs = loader.load()

    pages: List[Tuple[int, str]] = []
    total_chars = 0

    for idx, doc in enumerate(raw_docs):
        text = doc.page_content.strip()
        page_num = doc.metadata.get("page", idx)
        display_page = page_num + 1 if isinstance(page_num, int) else (idx + 1)
        if text:
            pages.append((display_page, text))
            total_chars += len(text)

    total_pdf_pages = len(raw_docs) if raw_docs else 0
    return pages, total_chars, total_pdf_pages


def _extract_ocr_pdf(temp_path: str, filename: str) -> Tuple[List[Tuple[int, str]], int, int]:
    """Extract text from scanned PDF pages using pypdfium2 and RapidOCR.

    Returns:
        List of (1-based page number, text content), total character count, and total page count.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is required for local PDF page rendering during OCR. "
            "Please install it with `pip install pypdfium2`."
        ) from exc

    ocr = _get_ocr_engine()

    try:
        pdf = pdfium.PdfDocument(temp_path)
    except Exception as exc:
        raise ValueError(f"Failed to render scanned PDF '{filename}': {exc}") from exc

    total_pdf_pages = len(pdf)
    if total_pdf_pages == 0:
        raise ValueError(f"Scanned PDF '{filename}' contains 0 pages.")

    pages: List[Tuple[int, str]] = []
    total_chars = 0

    for page_idx in range(total_pdf_pages):
        display_page = page_idx + 1
        page = pdf[page_idx]

        # Render page to bitmap at 2.0x scale (~144 dpi) for accurate text recognition
        bitmap = page.render(scale=2.0)
        img_np = bitmap.to_numpy()

        ocr_result, _ = ocr(img_np)
        if ocr_result:
            # RapidOCR returns list of [[box_coords], text, confidence]
            lines = [line[1].strip() for line in ocr_result if line and len(line) > 1 and line[1]]
            page_text = "\n".join(lines).strip()
            if page_text:
                pages.append((display_page, page_text))
                total_chars += len(page_text)

    return pages, total_chars, total_pdf_pages


def load_uploaded_document(uploaded_file) -> Tuple[List[Document], Dict[str, Any]]:
    """Load an uploaded file (PDF, TXT, or MD) into LangChain Documents with enriched metadata.

    Workflow:
    1. Validates file extension, non-empty payload, and computes SHA-256 hash.
    2. For PDFs:
       a. Attempts digital text extraction via PyPDFLoader.
       b. If digital text is empty (< 50 chars), automatically flags as scanned and triggers local OCR.
       c. If OCR detects text, builds Documents with 'Local OCR (RapidOCR)' metadata.
       d. If neither yields legible text, raises an informative, actionable error.
    3. For TXT and MD:
       a. Reads file content with UTF-8 encoding.
    4. Enriches every Document with page number, source, file type, size, and extraction method.
    """
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".txt", ".md"]:
        raise ValueError(f"Unsupported file format '{suffix}'. Please upload a .pdf, .txt, or .md file.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError(f"The file '{filename}' is completely empty (0 bytes).")

    file_size_kb = round(len(file_bytes) / 1024, 2)
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        enriched_docs: List[Document] = []
        extraction_method = "Digital (PyPDF)"
        total_pages = 1
        total_chars = 0

        if suffix == ".pdf":
            # Step 1: Attempt digital text extraction
            try:
                pages_data, total_chars, total_pages = _extract_digital_pdf(temp_path)
            except Exception:
                # If PyPDF fails to parse internal streams, fall back to OCR
                pages_data, total_chars, total_pages = [], 0, 0

            # Step 2: Automatic scanned detection & local OCR fallback
            if total_chars < 50:
                extraction_method = "Local OCR (RapidOCR)"
                try:
                    pages_data, total_chars, total_pages = _extract_ocr_pdf(temp_path, filename)
                except Exception as ocr_err:
                    raise ValueError(
                        f"The document '{filename}' contains no readable digital text, and local OCR failed: {ocr_err}"
                    ) from ocr_err

                if total_chars < 30:
                    raise ValueError(
                        f"The document '{filename}' appears to be a scanned image-only PDF, "
                        "but local OCR could not recognize any legible text. "
                        "Please verify the document contains clear, high-contrast text."
                    )
            else:
                extraction_method = "Digital Text (PyPDF)"

            # Build enriched LangChain Document objects page by page
            for page_num, text_content in pages_data:
                meta = {
                    "source": filename,
                    "page": page_num,
                    "file_type": "PDF",
                    "file_size_kb": file_size_kb,
                    "file_hash": file_hash,
                    "extraction_method": extraction_method,
                }
                enriched_docs.append(Document(page_content=text_content, metadata=meta))

        else:
            # TXT or Markdown file
            extraction_method = "Direct Text"
            loader = TextLoader(temp_path, encoding="utf-8")
            raw_docs = loader.load()

            if not raw_docs:
                raise ValueError(f"No content could be read from text document '{filename}'.")

            total_chars = 0
            for idx, doc in enumerate(raw_docs):
                cleaned = doc.page_content.strip()
                if cleaned:
                    meta = {
                        "source": filename,
                        "page": idx + 1,
                        "file_type": suffix.replace(".", "").upper(),
                        "file_size_kb": file_size_kb,
                        "file_hash": file_hash,
                        "extraction_method": extraction_method,
                    }
                    enriched_docs.append(Document(page_content=cleaned, metadata=meta))
                    total_chars += len(cleaned)

            if total_chars < 10:
                raise ValueError(f"The text document '{filename}' contains virtually no text ({total_chars} characters).")

            total_pages = len(enriched_docs)

        doc_meta = {
            "filename": filename,
            "file_type": suffix.replace(".", "").upper(),
            "pages": total_pages,
            "readable_pages": len(enriched_docs),
            "total_chars": total_chars,
            "file_size_kb": file_size_kb,
            "file_hash": file_hash,
            "extraction_method": extraction_method,
        }

        return enriched_docs, doc_meta

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def load_single_file(uploaded_file):
    """Alias for load_uploaded_document."""
    return load_uploaded_document(uploaded_file)


def load_multiple_documents(uploaded_files: List[Any]):
    """Alias for multiple files support."""
    if not uploaded_files:
        raise ValueError("No files provided.")
    return load_uploaded_document(uploaded_files[0])
