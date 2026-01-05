# relevance/relevant_snippets.py
from __future__ import annotations

import logging
from typing import List, Optional

from relevance_model import Document, Snippet

logger = logging.getLogger(__name__)


def find_relevant_snippet(
    doc: Document,
    relevant_snippet: Optional[str],
) -> Optional[List[Snippet]]:
    """
    Minimal version:
    - If LLM gives a snippet string, wrap it as Snippet(text=...)
    - No fuzzy matching, no char offsets
    - NEVER raises
    """
    if not relevant_snippet:
        return None

    try:
        return [Snippet(text=relevant_snippet)]
    except Exception as e:
        logger.exception("Failed to build relevant snippet: %s", e)
        return None
