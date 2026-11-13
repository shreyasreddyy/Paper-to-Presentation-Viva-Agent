"""Vector store management, document chunking, semantic similarity search, and keyword filtering with ChromaDB."""

import uuid
from typing import List, Tuple, Dict, Any
import chromadb
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, COLLECTION_NAME


def split_documents_into_chunks(docs: List[Document]) -> List[Document]:
    """Split documents using RecursiveCharacterTextSplitter and tag each chunk with rich metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    raw_chunks = splitter.split_documents(docs)
    enriched_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        meta = dict(chunk.metadata)
        meta["chunk_id"] = idx + 1
        meta["char_length"] = len(chunk.page_content)
        enriched_chunks.append(Document(page_content=chunk.page_content, metadata=meta))
    return enriched_chunks


def build_vector_store(docs: List[Document], embedder: Any) -> Tuple[Chroma, List[Document]]:
    """Build an in-memory ChromaDB vector store from document list.

    Returns:
        (Chroma vectorstore instance, list of indexed chunks)
    """
    chunks = split_documents_into_chunks(docs)
    if not chunks:
        raise ValueError("No readable text found in document.")

    # Use EphemeralClient for isolated in-memory vector storage per session
    client = chromadb.EphemeralClient()
    # Unique collection name to ensure clean state
    unique_col_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}"

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        client=client,
        collection_name=unique_col_name,
    )
    return store, chunks


def retrieve_context_with_metadata(
    store: Chroma, query: str, k: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    """Perform similarity search and return both a formatted prompt context and structured chunk metadata.

    Returns:
        (formatted_context_string, list_of_retrieved_chunk_info)
    """
    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception:
        raw_docs = store.similarity_search(query, k=k)
        results = [(d, 0.0) for d in raw_docs]

    retrieved_info = []
    context_blocks = []

    best_score = 999.0
    for rank, (doc, score) in enumerate(results, start=1):
        score_val = round(float(score), 4)
        if score_val < best_score:
            best_score = score_val

        source = doc.metadata.get("source", "Uploaded Document")
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", rank)

        info = {
            "rank": rank,
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "score": score_val,
            "content": doc.page_content,
        }
        retrieved_info.append(info)

        block = f"[Source: {source} | Page: {page} | Chunk: #{chunk_id} | Distance: {score_val}]\n{doc.page_content}"
        context_blocks.append(block)

    formatted_context = "\n\n---\n\n".join(context_blocks)
    if best_score > 0.95 and context_blocks:
        formatted_context = (
            f"[Grounding Notice: The closest retrieved document passage has a high semantic distance ({best_score}) "
            f"to the query '{query}'. If the document does not discuss this topic, report that the information is unavailable.]\n\n"
            + formatted_context
        )

    return formatted_context, retrieved_info


def search_chunks_by_keyword(chunks: List[Document], keyword: str) -> List[Document]:
    """Filter indexed chunks containing a specified keyword (case-insensitive)."""
    clean_kw = keyword.strip().lower()
    if not clean_kw:
        return chunks
    return [c for c in chunks if clean_kw in c.page_content.lower()]
