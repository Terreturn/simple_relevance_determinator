# main.py
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from LLM_calls import DeepSeekClient
from relevance_model import Document, RelevanceCriteria, RelevanceCriterion
from relevance_judge import judge_relevance
from tracing import setup_logging
import os

from eval_utils import (
    auc_roc,
    binary_metrics_from_scores,
    build_grouped_inputs,
    calibrate_threshold_grid_search,
    gold_to_binary,
    precision_at_k,
)


# ----------------------------
# Data loading helpers
# ----------------------------

def load_documents_from_jsonl(path: str) -> List[Document]:
    """
    JSONL line format (each line is a JSON object):
    {
      "doc_id": "xxx" or "corpus_id": "xxx",
      "title": "...",
      "abstract": "...",
      "markdown": "..."
    }
    """
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            doc_id = obj.get("corpus_id") or obj.get("doc_id")
            if not doc_id:
                raise ValueError("Each doc jsonl row must contain 'corpus_id' or 'doc_id'.")

            markdown = obj.get("markdown") or ""
            docs.append(
                Document(
                    corpus_id=str(doc_id),
                    title=obj.get("title"),
                    abstract=obj.get("abstract"),
                    markdown=str(markdown),
                )
            )
    return docs


def load_eval_rows(path: str) -> List[Dict[str, object]]:
    """
    JSONL format:
      {"query": "...", "doc_id": "...", "gold_label": 0/1/2/3}
    """
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "query" not in obj or "doc_id" not in obj or "gold_label" not in obj:
                raise ValueError("Eval row must contain query, doc_id, gold_label.")
            rows.append(obj)
    return rows


def demo_documents() -> List[Document]:
    return [
        Document(
            corpus_id="d1",
            title="Graph Neural Networks for Traffic Forecasting",
            abstract="We propose a GNN model for traffic speed prediction using road network graphs.",
            markdown="This paper introduces a graph neural network approach for traffic forecasting ...",
        ),
        Document(
            corpus_id="d2",
            title="A Survey on Image Classification",
            abstract="We survey CNN-based image classification methods.",
            markdown="This survey covers convolutional neural networks for image classification ...",
        ),
    ]


# ----------------------------
# Output helpers
# ----------------------------

def print_results(results, top_n: int = 20) -> None:
    results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
    print("\n=== Relevance results (sorted by score) ===")
    for r in results[:top_n]:
        summary = (r.relevance_summary or "").replace("\n", " ")
        if len(summary) > 140:
            summary = summary[:140] + "..."
        print(
            f"- doc_id={r.doc_id} score={r.relevance_score:.3f} level={r.relevance} "
            f"model={r.relevance_model_name} summary={summary}"
        )


def write_results_jsonl(results, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "doc_id": r.doc_id,
                "relevance_score": r.relevance_score,
                "relevance_level": r.relevance,
                "model_name": r.relevance_model_name,
                "relevance_summary": r.relevance_summary,
                "criteria_judgements": [
                    {
                        "name": cj.name,
                        "relevance": cj.relevance,
                        "relevant_snippets": [
                            getattr(s, "text", str(s)) for s in (cj.relevant_snippets or [])
                        ]
                        if cj.relevant_snippets
                        else None,
                    }
                    for cj in r.relevance_criteria_judgements
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# Criteria builder
# ----------------------------

def build_criteria(query: str) -> RelevanceCriteria:

    criteria = [
        RelevanceCriterion(
            name="Directly answers the question",
            description="The paper directly addresses the user's question or provides methods/results that solve it.",
            weight=0.7,
            nice_to_have=False,
        ),
        RelevanceCriterion(
            name="Provides useful methods or evidence",
            description="The paper provides methods, experiments, datasets, or evidence that are useful for the question.",
            weight=0.3,
            nice_to_have=False,
        ),
    ]
    return RelevanceCriteria(query=query, criteria=criteria)


# ----------------------------
# Evaluation runner (optional)
# ----------------------------

def run_evaluation(
    eval_rows: List[Dict[str, object]],
    results_by_query_doc: Dict[Tuple[str, str], float],
    positive_from: int = 1,
    k: int = 5,
) -> None:
    """
    eval_rows: [{"query","doc_id","gold_label"}...]
    results_by_query_doc: {(query, doc_id): relevance_score}
    """
    flat_scores, flat_gold_0to3, results_by_query, gold_by_query = build_grouped_inputs(
        eval_rows,
        results_by_query_doc,
    )
    if not flat_scores:
        print("\n[Eval] No matched (query,doc) scores found. Skipping evaluation.")
        return

    gold_bin = [gold_to_binary(int(g), positive_from=positive_from) for g in flat_gold_0to3]

    # 1) threshold calibration
    best = calibrate_threshold_grid_search(flat_scores, gold_bin, steps=50, metric="f1")

    # 2) metrics at best threshold
    bm = binary_metrics_from_scores(flat_scores, gold_bin, best.threshold)

    # 3) AUROC
    auc = auc_roc(flat_scores, gold_bin)

    # 4) Precision@K
    p_at_k = precision_at_k(results_by_query, gold_by_query, k=k, positive_from=positive_from)

    print("\n=== Evaluation ===")
    print(f"[Calibration] best_threshold={best.threshold:.2f}  (target=f1)")
    print(f"[Binary] precision={bm.precision:.3f} recall={bm.recall:.3f} accuracy={bm.accuracy:.3f} f1={bm.f1:.3f}")
    print(f"[AUROC] {auc if auc is not None else 'N/A (only one class)'}")
    print(f"[Precision@{k}] {p_at_k:.3f}")


# ----------------------------
# Main
# ----------------------------

async def async_main(args) -> int:
    setup_logging()

    # LLM client (config-driven)
    llm = DeepSeekClient(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
)

    # documents
    if args.docs_jsonl:
        documents = load_documents_from_jsonl(args.docs_jsonl)
    else:
        documents = demo_documents()

    # criteria
    relevance_criteria = build_criteria(args.query)

    # judge
    results = await judge_relevance(documents, relevance_criteria, llm)

    # show
    print_results(results, top_n=args.top_n)

    # save
    if args.out_jsonl:
        write_results_jsonl(results, args.out_jsonl)
        print(f"\nWrote results to: {args.out_jsonl}")

    # optional evaluation
    if args.eval_jsonl:
        eval_rows = load_eval_rows(args.eval_jsonl)

        # Build score map keyed by (query, doc_id)
        score_by_key: Dict[Tuple[str, str], float] = {}
        score_by_doc: Dict[str, float] = {r.doc_id: r.relevance_score for r in results}

        for row in eval_rows:
            q = str(row["query"])
            doc_id = str(row["doc_id"])
            if q != args.query:
                continue
            if doc_id in score_by_doc:
                score_by_key[(q, doc_id)] = float(score_by_doc[doc_id])

        run_evaluation(
            eval_rows=eval_rows,
            results_by_query_doc=score_by_key,
            positive_from=args.positive_from,
            k=args.k,
        )

    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser("paper relevance judge")

    p.add_argument("--query", type=str, required=True, help="user question / query")
    p.add_argument("--docs-jsonl", type=str, default=None, help="path to documents jsonl")
    p.add_argument("--out-jsonl", type=str, default=None, help="write results jsonl")
    p.add_argument("--top-n", type=int, default=20, help="print top N results")

    # eval
    p.add_argument("--eval-jsonl", type=str, default=None, help="path to eval jsonl (query,doc_id,gold_label)")
    p.add_argument("--positive-from", type=int, default=1, choices=[1, 2, 3], help="gold_label >= this => positive")
    p.add_argument("--k", type=int, default=5, help="K for Precision@K")

    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
