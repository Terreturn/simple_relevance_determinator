# relevance/judge.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, ValidationError, create_model

from config import CONFIG
from LLM_calls import DeepSeekClient, LLMResponse
from relevance_model import (
    Document,
    RelevanceCriteria,
    RelevanceCriterion,
    RelevanceCriterionJudgement,
    RelevanceJudgement,
    RELEVANCE_LABEL_TO_SCORE,
    RelevanceThresholds,
)
from relevance_prompts import criteria_to_json_payload, render_relevance_prompt
from text_cutting import build_llm_document_input

from relevant_snippets import find_relevant_snippet
HAVE_SNIPPET_LOCATOR = True

HAVE_SNIPPET_LOCATOR = False  # False for now

logger = logging.getLogger(__name__)


# --------- 1) schma ---------

# Dynamically create pydantic models for LLM output parsing/validation
class _CriterionValueModel(BaseModel):
    relevance: str
    relevant_snippet: Optional[str] = None


def _create_dynamic_output_model(criteria: List[RelevanceCriterion]) -> Type[BaseModel]:
    fields: Dict[str, Any] = {c.name: (_CriterionValueModel, ...) for c in criteria}
    criteria_model = create_model("CriteriaResult", **fields)

    output_model = create_model(
        "RelevanceJudgementResult",
        criteria=(criteria_model, ...),
        relevance_summary=(Optional[str], ...),
    )
    return output_model


# --------- 2) calculate score ---------

def _calculate_weighted_score(
    relevance_criteria: RelevanceCriteria,
    judgements: List[RelevanceCriterionJudgement],
) -> float:
    name_to_weight: Dict[str, float] = {}
    for c in relevance_criteria.to_flat_criteria(include_nice_to_have=True):
        name_to_weight[c.name] = c.weight

    score = 0.0
    for j in judgements:
        w = name_to_weight.get(j.name, 0.0)
        score += w * (float(j.relevance) / 3.0)

    return min(1.0, score)


def _convert_score_to_level(score: float) -> int:
    if score <= RelevanceThresholds.NOT_RELEVANT:
        return 0
    elif score <= RelevanceThresholds.SOMEWHAT_RELEVANT:
        return 1
    elif score <= RelevanceThresholds.HIGHLY_RELEVANT:
        return 2
    return 3


# --------- 3) single paper：LLM -> judgements -> RelevanceJudgement ---------

def _build_criterion_judgements(
    doc: Document,
    criteria_output: Dict[str, Dict[str, Any]],
) -> List[RelevanceCriterionJudgement]:
    """
    criteria_output: {criterion_name: {"relevance": "...", "relevant_snippet": "..."/None}}
    """
    out: List[RelevanceCriterionJudgement] = []

    for name, val in criteria_output.items():
        label = (val.get("relevance") or "").strip()
        if label not in RELEVANCE_LABEL_TO_SCORE:
            logger.warning("Unknown relevance label=%r for criterion=%r; fallback to Not Relevant", label, name)
            rel_num = RELEVANCE_LABEL_TO_SCORE["Not Relevant"]
        else:
            rel_num = RELEVANCE_LABEL_TO_SCORE[label]

        snippet_text = val.get("relevant_snippet")

        # locate snippet
        relevant_snippets = None
        if HAVE_SNIPPET_LOCATOR:
            try:
                relevant_snippets = find_relevant_snippet(doc, snippet_text)
            except Exception as e:
                logger.exception("find_relevant_snippet failed: %s", e)
                relevant_snippets = None
        out.append(
            RelevanceCriterionJudgement(
                name=name,
                relevance=int(rel_num),
                relevant_snippets=relevant_snippets,
            )
        )

    return out


async def _judge_one_document(
    doc: Document,
    relevance_criteria: RelevanceCriteria,
    criteria_list: List[RelevanceCriterion],
    llm: DeepSeekClient,
    output_model: Type[BaseModel],
) -> RelevanceJudgement:
    # 1) input 
    doc_text = build_llm_document_input(doc, max_chars=CONFIG.llm.max_input_chars)

    # 2) prompt construction
    criteria_json = criteria_to_json_payload(criteria_list)
    prompt = render_relevance_prompt(criteria_json)

    # 3) LLM call + parse
    parsed, resp = await llm.call_pydantic(
        instructions=prompt,
        user_input=doc_text,
        output_model=output_model,
        temperature=CONFIG.llm.temperature,
    )

    data = parsed.model_dump()
    criteria_output = data["criteria"]  # dict
    relevance_summary = data.get("relevance_summary")

    # 4) build criterion judgements
    criterion_judgements = _build_criterion_judgements(doc, criteria_output)

    # 5) check all required criteria are judged
    required_names = [c.name for c in relevance_criteria.to_flat_criteria(include_nice_to_have=False)]
    judged_names = [cj.name for cj in criterion_judgements]
    missing = [n for n in required_names if n not in judged_names]
    if missing:
        # missing required criteria
        raise ValueError(f"Missing required criteria in LLM output: {missing}")

    # 6) aggregate score + level
    score = _calculate_weighted_score(relevance_criteria, criterion_judgements)
    level = _convert_score_to_level(score)

    return RelevanceJudgement(
        doc_id=doc.corpus_id,
        relevance=level,
        relevance_score=score,
        relevance_model_name=resp.model_name,
        relevance_criteria_judgements=criterion_judgements,
        relevance_summary=relevance_summary,
        debug={
            "input_chars": len(doc_text),
            "usage": resp.usage,
        },
    )


# --------- 4) output api ---------

async def judge_relevance(
    documents: Sequence[Document],
    relevance_criteria: RelevanceCriteria,
    llm: DeepSeekClient,
) -> List[RelevanceJudgement]:
    """
    Main entrypoint you will call from main.py.

    - Concurrency controlled here (doc-level)
    - LLM retry/timeout handled inside llm client as well
    """
    criteria_list = relevance_criteria.to_flat_criteria(include_nice_to_have=False)
    if not criteria_list:
        return []

    output_model = _create_dynamic_output_model(criteria_list)

    sem = asyncio.Semaphore(CONFIG.llm.concurrency)

    async def _run(doc: Document) -> RelevanceJudgement:
        async with sem:
            return await _judge_one_document(doc, relevance_criteria, criteria_list, llm, output_model)

    tasks = [_run(d) for d in documents]
    return await asyncio.gather(*tasks)
