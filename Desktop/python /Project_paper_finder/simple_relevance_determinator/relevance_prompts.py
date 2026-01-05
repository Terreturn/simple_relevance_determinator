# relevance/prompts.py
from __future__ import annotations

from typing import List

from relevance_model import RelevanceCriterion


RELEVANCE_JUDGEMENT_PROMPT_WITH_RELEVANT_SNIPPETS = """
Judge how relevant the following paper is to each of the provided criteria. For each criterion, consider its entire description when making your judgement.

For each criterion, provide the following outputs:
- `relevance` (str): one of "Perfectly Relevant", "Somewhat Relevant", "Not Relevant".
- `relevant_snippet` (str | null): a snippet from the document that best show the relevance of the paper to the criterion. To be clear, copy EXACT text ONLY. Choose one short text span that best shows the relevance in a concrete and specific way, up to 20 words. ONLY IF NECESSARY, you can add another few-words-long span (e.g. for coreference, disambiguation, necessary context), separated by ` ... `. If relevance is "Not Relevant" output null. The snippet may contain citations, but make sure to only take snippets that directly show the relevance of this paper. 

Also provide another field `relevance_summary` (str): a short summary explanation of how the paper is relevant to the criteria in general. null if it is not relevant to any of the criteria.
- This should be short but convey the most useful and specific information for a user skimming a list of papers, up to 30 words.
- No need to mention which are perfectly relevant, somewhat relevant, or not relevant. Just provide new information that was not mentioned in the criteria.
- Start with perfectly relevant ones, include a specific and interesting detail about what matches them. Then go on to somewhat relevant ones, saying why it is close but not a perfect match. No need to add extra info for not relevant ones.
- Start with actionable info. Instead of saying "The paper uses X to...", just say "Uses X to...".

Output a JSON:
- top-level key `criteria`. Under it, for every criterion name (exactly as given in the provided criteria), there should be an object containing two fields: `relevance` and `relevant_snippet`.
- top-level key `relevance_summary` with string value or null.


Criteria:
```
{{{criteria}}}
```"""

## ----------------------------
## Helpers

def render_relevance_prompt(criteria_json: str) -> str:
    """
    Substitute the mustache placeholder {{{criteria}}}.
    """
    return RELEVANCE_JUDGEMENT_PROMPT_WITH_RELEVANT_SNIPPETS.replace("{{{criteria}}}", criteria_json)


def criteria_to_json_payload(criteria: List[RelevanceCriterion]) -> str:
    """
    The original repo used json.dumps([...], indent=2). We'll keep this helper here so judge.py stays clean.
    """
    import json

    return json.dumps(
        [{"name": c.name, "description": c.description} for c in criteria],
        ensure_ascii=False,
        indent=2,
    )
