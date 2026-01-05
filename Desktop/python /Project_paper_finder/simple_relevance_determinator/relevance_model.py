from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Union


# ---------- Core document structures ----------

CorpusId = str


@dataclass
class Snippet:
    """
    A text span from the document, optionally anchored by offsets for highlighting.
    Offsets should be in the coordinate system of the source document text that
    snippet was extracted from (you decide: full markdown or section text).
    """
    text: str
    section_kind: Optional[str] = None
    section_title: Optional[str] = None
    char_start_offset: Optional[int] = None
    char_end_offset: Optional[int] = None


@dataclass
class CitationContext:
    """
    Minimal citation context. In the original repo it can be used as a source of text spans.
    """
    text: str
    source_corpus_id: Optional[CorpusId] = None



@dataclass
class Document:
    """
    Minimal Document that matches what your relevance code expects.
    """
    corpus_id: str
    markdown: str

    title: Optional[str] = None
    abstract: Optional[str] = None

    # Optional: structured text chunks for snippet matching / UI highlighting
    snippets: List["Snippet"] = field(default_factory=list)
    citation_contexts: List["CitationContext"] = field(default_factory=list)


# ---------- Relevance criteria ----------

@dataclass
class RelevanceCriterion:
    """
    One criterion the LLM judges against, with a weight used for aggregation.
    """
    name: str
    description: str
    weight: float = 1.0
    nice_to_have: bool = False  # to support include_nice_to_have flag


@dataclass
class RelevanceCriteria:
    """
    Holds the query and the list of criteria.
    """
    query: str
    criteria: list[RelevanceCriterion]

    def to_flat_criteria(self, include_nice_to_have: bool = False) -> list[RelevanceCriterion]:
        if include_nice_to_have:
            return list(self.criteria)
        return [c for c in self.criteria if not c.nice_to_have]


# ---------- Judgement output models ----------

RelevanceLabel = Literal["Perfectly Relevant", "Somewhat Relevant", "Not Relevant"]

RELEVANCE_LABEL_TO_SCORE: dict[str, int] = {
    "Perfectly Relevant": 3,
    "Somewhat Relevant": 1,
    "Not Relevant": 0,
}


@dataclass
class RelevanceCriterionJudgement:
    """
    A judgement for a single criterion for a given document.
    `relevance` is numeric in 0..3 to align with the original repo aggregation (/3).
    """
    name: str
    relevance: int  # 0..3
    relevant_snippets: Optional[list[Snippet | CitationContext]] = None


@dataclass
class RelevanceJudgement:
    """
    Final per-document judgement: overall score + level + evidence.
    - relevance_score: continuous 0..1
    - relevance: discrete 0..3 level (after thresholding)
    """
    doc_id: CorpusId
    relevance: int
    relevance_score: float

    relevance_model_name: str
    relevance_criteria_judgements: list[RelevanceCriterionJudgement] = field(default_factory=list)
    relevance_summary: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None

    # optional debugging payload (raw llm JSON, timings, etc.)
    # debug: dict[str, Any] = field(default_factory=dict)


# ---------- Thresholds ----------

@dataclass
class RelevanceThresholds:
    NOT_RELEVANT = 0.25
    SOMEWHAT_RELEVANT = 0.67
    HIGHLY_RELEVANT = 0.99
