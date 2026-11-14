"""Intent detection and conditional retrieval router using LangChain Prompt Templates and LCEL."""

import json
import re
from typing import Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


ROUTER_SYSTEM_PROMPT = """You are an intent-routing classifier for an academic research assistant.
Analyze the user request and classify it into exactly one of the supported specialist intents, and determine whether retrieval from the research paper is needed.

Supported intents:
- summary: User wants a high-level summary, executive abstract, or section-by-section digest of the paper.
- concepts: User wants key technical concepts, terms, definitions, formulas, or method explanations.
- slides: User wants presentation slides, talk outlines, or defense presentation decks.
- viva: User wants oral exam / viva voce / thesis defense questions and answers.
- explain: User is asking a specific factual question or seeking clarification on a paper topic.

Retrieval rule:
- retrieve = true: ALWAYS set retrieve=true whenever the request asks for a summary, key concepts, presentation slides, viva questions, or details about the uploaded paper.
- retrieve = false: Set retrieve=false ONLY if the request is a general greeting or an abstract query entirely independent of any document.

Few-shot examples:
Example 1:
User request: "Prepare 8-10 slides for my presentation."
Output:
{{"intent": "slides", "retrieve": true, "reason": "Presentation slides require technical content and results from the uploaded paper."}}

Example 2:
User request: "Generate 10 viva defense questions."
Output:
{{"intent": "viva", "retrieve": true, "reason": "Viva questions must be grounded in the paper methodology and findings."}}

Example 3:
User request: "Summarize the paper."
Output:
{{"intent": "summary", "retrieve": true, "reason": "Summarization requires retrieving the core sections of the uploaded document."}}

Example 4:
User request: "What does this application do?"
Output:
{{"intent": "explain", "retrieve": false, "reason": "Meta inquiry about the tool, independent of document content."}}

Output Format:
You MUST respond ONLY with a single valid JSON object containing exactly the keys "intent", "retrieve", and "reason". No introductory text or markdown formatting outside the JSON object.
"""

ROUTER_USER_PROMPT = "User request: {query}"


def get_router_chain(llm):
    """Create an LCEL intent routing chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_USER_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def safe_route(query: str, llm, has_document: bool = True) -> Tuple[Dict[str, Any], str]:
    """Execute intent routing with JSON parsing, validation, and deterministic paper-retrieval enforcement.

    Args:
        query: User input query
        llm: ChatOllama instance
        has_document: Whether an active document index exists in ChromaDB

    Returns:
        (parsed_dict, raw_llm_output)
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "intent": "summary",
            "retrieve": has_document,
            "reason": "Defaulted to summary for empty request." if has_document else "No document indexed; retrieval skipped.",
        }, "{}"

    chain = get_router_chain(llm)
    try:
        raw_output = chain.invoke({"query": clean_query})
    except Exception as e:
        raw_output = str(e)

    # Try extracting JSON object
    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = json.loads(raw_output)

        intent = str(parsed.get("intent", "")).lower().strip()
        if intent not in {"summary", "concepts", "slides", "viva", "explain"}:
            raise ValueError(f"Unknown intent: {intent}")

        # Mandatory rule: If a paper is loaded and the intent is summary, concepts, slides, or viva,
        # retrieval MUST be True.
        if has_document:
            if intent in {"summary", "concepts", "slides", "viva"}:
                retrieve = True
                reason = str(parsed.get("reason", f"Retrieval required for {intent} generation from paper context."))
            else:
                retrieve = bool(parsed.get("retrieve", True))
                reason = str(parsed.get("reason", "Determined by LLM classifier."))
        else:
            retrieve = False
            reason = "Retrieval was skipped because no document is currently indexed in the vector store."

        return {
            "intent": intent,
            "retrieve": retrieve,
            "reason": reason,
        }, raw_output

    except Exception:
        # Safe deterministic keyword fallback
        q_lower = clean_query.lower()
        if any(w in q_lower for w in ["viva", "exam", "question", "defense", "examiner"]):
            fallback_intent = "viva"
        elif any(w in q_lower for w in ["slide", "presentation", "deck", "powerpoint", "pptx"]):
            fallback_intent = "slides"
        elif any(w in q_lower for w in ["concept", "term", "formula", "definition", "algorithm"]):
            fallback_intent = "concepts"
        elif any(w in q_lower for w in ["explain", "why", "how", "what is", "detail"]):
            fallback_intent = "explain"
        else:
            fallback_intent = "summary"

        return {
            "intent": fallback_intent,
            "retrieve": has_document,
            "reason": "Heuristic fallback routing applied.",
        }, raw_output
