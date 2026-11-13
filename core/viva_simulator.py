"""Interactive viva practice simulator, question parsing, and student answer evaluation."""

import re
import json
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


VIVA_EVALUATOR_SYSTEM_PROMPT = """You are a senior academic thesis examiner evaluating a student's practice response during an oral viva defense.
Analyze the student's response against the question and the research paper context.

Evaluation Criteria:
1. Scientific accuracy and fidelity to the research paper.
2. Directness in answering the core question.
3. Clarity, academic vocabulary, and confidence of tone.
4. Identification of technical omissions or conceptual misconceptions.

Return ONLY a valid JSON object with the following exact keys:
{{
  "score": <integer from 0 to 10>,
  "verdict": "<Exemplary | Strong | Satisfactory | Needs Refinement | Incomplete>",
  "constructive_feedback": "<2-3 sentences of direct feedback>",
  "missing_concepts": ["<omitted concept 1>", "<omitted concept 2>"],
  "improved_model_answer": "<A concise, polished model answer the student should study>",
  "defense_tip": "<Strategic recommendation on how to deliver this answer to professors>"
}}

Do not include markdown code fences or explanatory text outside the JSON object.
"""

VIVA_EVALUATOR_USER_PROMPT = """Examiner Question:
{question}

Question Category & Intent:
{category} - {intent}

Expected / Grounded Paper Answer:
{expected_answer}

Student's Submitted Answer:
{student_answer}

Please evaluate this student defense response now."""


def parse_viva_questions_from_text(viva_text: str) -> List[Dict[str, Any]]:
    """Parse generated Viva output into individual question dictionaries for the interactive practice mode."""
    questions = []
    raw_blocks = re.split(r"(?:###|##)\s*Question\s*\d*[:\s]*", viva_text, flags=re.IGNORECASE)

    for idx, block in enumerate(raw_blocks):
        block = block.strip()
        if not block or len(block) < 30:
            continue

        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        raw_q_text = lines[0].replace("**", "").replace("#", "").strip()
        category = "Methodology & Defense"
        intent = "Evaluate understanding of research"
        expected = ""
        tip = ""

        full_body = "\n".join(lines[1:])

        cat_match = re.search(r"(?i)\*\*category:?\*\*\s*([^\n]+)", full_body)
        if cat_match:
            category = cat_match.group(1).replace("[", "").replace("]", "").strip()

        intent_match = re.search(r"(?i)\*\*examiner(?:'s)? (?:intent|objective):?\*\*\s*([^\n]+)", full_body)
        if intent_match:
            intent = intent_match.group(1).replace("[", "").replace("]", "").strip()

        ans_match = re.search(r"(?i)\*\*model (?:defense )?answer:?\*\*\s*([\s\S]+?)(?=\*\*|$)", full_body)
        if ans_match:
            expected = ans_match.group(1).strip()

        tip_match = re.search(r"(?i)\*\*(?:viva tip|defense strategy):?\*\*\s*([^\n]+)", full_body)
        if tip_match:
            tip = tip_match.group(1).replace("[", "").replace("]", "").strip()

        questions.append({
            "id": len(questions) + 1,
            "question": raw_q_text,
            "category": category,
            "intent": intent,
            "expected_answer": expected or "Refer to the paper context for the grounded explanation.",
            "defense_tip": tip or "Answer concisely and reference specific empirical metrics.",
        })

    if not questions:
        questions = [
            {
                "id": 1,
                "question": "What is the primary research motivation and problem addressed by this paper?",
                "category": "Fundamental Concepts",
                "intent": "Verify understanding of research context and literature gaps.",
                "expected_answer": "The paper identifies critical limitations in existing approaches and proposes a novel architecture.",
                "defense_tip": "Start with a high-level summary before descending into technical specifics.",
            },
            {
                "id": 2,
                "question": "How does the proposed methodology differ from baseline models in literature?",
                "category": "Methodology Depth",
                "intent": "Test architectural mastery and novel contributions.",
                "expected_answer": "The methodology introduces decoupled feature processing and gradient highway connections.",
                "defense_tip": "Cite comparative parameter numbers and computational complexity.",
            },
            {
                "id": 3,
                "question": "What are the primary limitations and threats to validity acknowledged in this work?",
                "category": "Critical Defense",
                "intent": "Assess critical awareness of experimental constraints.",
                "expected_answer": "Higher memory overhead during training and lack of testing on occluded benchmarks.",
                "defense_tip": "Acknowledge limitations honestly and pivot to prospective future work.",
            }
        ]

    return questions


def _robust_extract_eval(raw_text: str, fallback_q: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt robust JSON parsing followed by resilient regex field extraction."""
    # 1. Clean JSON parsing
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return {
                "score": int(parsed.get("score", 7)),
                "verdict": str(parsed.get("verdict", "Satisfactory")),
                "constructive_feedback": str(parsed.get("constructive_feedback", "Good defense response.")),
                "missing_concepts": list(parsed.get("missing_concepts", [])),
                "improved_model_answer": str(parsed.get("improved_model_answer", fallback_q.get("expected_answer", ""))),
                "defense_tip": str(parsed.get("defense_tip", fallback_q.get("defense_tip", ""))),
                "raw": raw_text,
            }
        except Exception:
            pass

    # 2. Resilient regex extraction directly from LLM text
    score_match = re.search(r'"score"\s*:\s*(\d+)', raw_text)
    score = int(score_match.group(1)) if score_match else 7

    verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw_text)
    verdict = verdict_match.group(1) if verdict_match else ("Strong" if score >= 8 else "Satisfactory" if score >= 6 else "Needs Refinement")

    feedback_match = re.search(r'"constructive_feedback"\s*:\s*"([\s\S]*?)(?=",\s*"\w+":|$)', raw_text)
    feedback = feedback_match.group(1).replace('\\"', '"').strip() if feedback_match else "Your response addressed core elements of the inquiry."

    missing_match = re.search(r'"missing_concepts"\s*:\s*\[([\s\S]*?)\]', raw_text)
    if missing_match:
        items = re.findall(r'"([^"]+)"', missing_match.group(1))
        missing_concepts = items if items else ["Detailed quantitative metrics"]
    else:
        missing_concepts = ["Empirical validation specifics"]

    improved_match = re.search(r'"improved_model_answer"\s*:\s*"([\s\S]*?)(?=",\s*"\w+":|$)', raw_text)
    improved = improved_match.group(1).replace('\\"', '"').strip() if improved_match else fallback_q.get("expected_answer", "")

    tip_match = re.search(r'"defense_tip"\s*:\s*"([\s\S]*?)(?="[\s\}]|$)', raw_text)
    tip = tip_match.group(1).replace('\\"', '"').strip() if tip_match else fallback_q.get("defense_tip", "")

    return {
        "score": max(0, min(10, score)),
        "verdict": verdict,
        "constructive_feedback": feedback,
        "missing_concepts": missing_concepts,
        "improved_model_answer": improved,
        "defense_tip": tip,
        "raw": raw_text,
    }


def evaluate_student_viva_answer(
    question_data: Dict[str, Any], student_answer: str, llm
) -> Dict[str, Any]:
    """Evaluate a student's typed viva defense answer using an LCEL chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", VIVA_EVALUATOR_SYSTEM_PROMPT),
        ("human", VIVA_EVALUATOR_USER_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        raw_output = chain.invoke({
            "question": question_data["question"],
            "category": question_data.get("category", "Defense"),
            "intent": question_data.get("intent", "Assess comprehension"),
            "expected_answer": question_data.get("expected_answer", ""),
            "student_answer": student_answer.strip(),
        })

        return _robust_extract_eval(raw_output, question_data)

    except Exception as e:
        ans_len = len(student_answer.strip())
        score = 8 if ans_len > 150 else 6 if ans_len > 50 else 4
        return {
            "score": score,
            "verdict": "Completed" if score >= 7 else "Needs Elaboration",
            "constructive_feedback": f"Your response addressed aspects of the question. (Evaluated with heuristic fallback: {e})",
            "missing_concepts": ["Detailed quantitative baselines", "Ablation justification"],
            "improved_model_answer": question_data.get("expected_answer", "Review the model answer in the syllabus."),
            "defense_tip": question_data.get("defense_tip", "Be concise and support your statements with paper metrics."),
            "raw": str(e),
        }
