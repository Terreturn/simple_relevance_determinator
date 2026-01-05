# text_cutting.py

from __future__ import annotations

import re
from typing import Iterable, Optional

from relevance_model import Document, Snippet  


# ---------- Text cutting and LLM input building ----------
def _clean_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")  # nbsp
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < int(0.6 * max_chars):
        cut = max_chars
    return text[:cut].rstrip() + "\n\n[TRUNCATED]"


def build_llm_document_input(
    doc: Document,
    *,
    max_chars: int = 12000,
    max_snippets: int = 8,
    snippet_chars: int = 600,
    include_markdown_tail: bool = False,
) -> str:
    """
    Build a compact document string for LLM judging relevance.

    Strategy:
    1) Always include title + abstract (high signal, short)
    2) Include up to `max_snippets` snippet texts (if available)
    3) Include a truncated slice of markdown to support snippet evidence when needed
       - default takes the beginning; optional tail helps when key info is near the end
    """

    parts: list[str] = []

    if doc.title:
        parts.append(f"# Title\n{_clean_whitespace(doc.title)}")
    if doc.abstract:
        parts.append(f"# Abstract\n{_clean_whitespace(doc.abstract)}")

    # Add snippets if present (often from retrieval; high relevance signal)
    if doc.snippets:
        picked = doc.snippets[:max_snippets]
        sn_texts = []
        for i, sn in enumerate(picked, start=1):
            t = _clean_whitespace(sn.text)
            t = _safe_truncate_chars(t, snippet_chars)
            sn_texts.append(f"[Snippet {i}]\n{t}")
        parts.append("# Key Snippets\n" + "\n\n".join(sn_texts))

    # Add markdown excerpt (for evidence). Keep it bounded.
    # If you already have strong snippets, you can reduce markdown budget a lot.
    remaining_budget = max_chars - sum(len(p) for p in parts) - 200  # buffer for headers/newlines
    if remaining_budget > 800 and doc.markdown:
        md = _clean_whitespace(doc.markdown)

        if include_markdown_tail and len(md) > remaining_budget:
            head_budget = int(0.7 * remaining_budget)
            tail_budget = remaining_budget - head_budget

            head = _safe_truncate_chars(md, head_budget)
            tail = md[-tail_budget:]
            tail = _safe_truncate_chars(tail, tail_budget)
            parts.append("# Paper (Excerpt)\n" + head + "\n\n...\n\n" + tail)
        else:
            parts.append("# Paper (Excerpt)\n" + _safe_truncate_chars(md, remaining_budget))

    final_text = "\n\n".join(parts)
    return _safe_truncate_chars(final_text, max_chars)
