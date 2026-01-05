# eval/eval_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ThresholdSearchResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    accuracy: float
    f1: float


def gold_to_binary(gold_label: int, positive_from: int = 1) -> int:
    """
    Convert 0..3 label into binary {0,1}.
    - positive_from=1: {1,2,3} are positive
    - positive_from=2: {2,3} are positive
    """
    return 1 if gold_label >= positive_from else 0


def binary_metrics_from_scores(
    scores: List[float],
    gold: List[int],
    threshold: float,
) -> BinaryMetrics:
    """
    Compute precision/recall/accuracy/f1 for binary classification:
        pred = 1 if score >= threshold else 0
    """
    assert len(scores) == len(gold)

    tp = fp = fn = tn = 0
    for s, g in zip(scores, gold):
        p = 1 if s >= threshold else 0
        if p == 1 and g == 1:
            tp += 1
        elif p == 1 and g == 0:
            fp += 1
        elif p == 0 and g == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return BinaryMetrics(
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
    )


def calibrate_threshold_grid_search(
    scores: List[float],
    gold: List[int],
    steps: int = 50,
    metric: str = "f1",
) -> ThresholdSearchResult:
    """
    Grid-search threshold in [0,1] with `steps` points.
    metric: "f1" (default) or "precision" or "recall" or "accuracy"
    """
    assert metric in {"f1", "precision", "recall", "accuracy"}
    assert len(scores) == len(gold)

    best: Optional[ThresholdSearchResult] = None

    # steps=50 -> 0.00, 0.02, ..., 1.00
    for i in range(steps + 1):
        t = i / float(steps)
        m = binary_metrics_from_scores(scores, gold, t)

        score = getattr(m, metric)
        cand = ThresholdSearchResult(
            threshold=t,
            f1=m.f1,
            precision=m.precision,
            recall=m.recall,
            accuracy=m.accuracy,
        )

        if best is None:
            best = cand
        else:
            # choose best by target metric; tie-break by higher precision (often better UX for paper finder)
            best_score = getattr(best, metric)
            if (score > best_score) or (abs(score - best_score) < 1e-12 and cand.precision > best.precision):
                best = cand

    assert best is not None
    return best


def auc_roc(scores: List[float], labels: List[int]) -> Optional[float]:
    """
    AUROC without sklearn: pairwise ranking method.
    Returns None if only one class present.
    """
    assert len(scores) == len(labels)

    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None

    correct = 0.0
    total = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                correct += 1.0
            elif sp == sn:
                correct += 0.5
            total += 1.0

    return correct / (total + 1e-9)


def precision_at_k(
    results_by_query: Dict[str, List[Tuple[str, float]]],
    gold_by_query: Dict[str, Dict[str, int]],
    k: int = 5,
    positive_from: int = 1,
) -> float:
    """
    results_by_query[q] = [(doc_id, score), ...]
    gold_by_query[q][doc_id] = gold_label (0..3)

    Precision@K averaged over queries.
    """
    if not results_by_query:
        return 0.0

    per_q: List[float] = []
    for q, items in results_by_query.items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:k]
        gold_map = gold_by_query.get(q, {})

        hits = 0
        for doc_id, _ in ranked:
            gl = gold_map.get(doc_id, 0)
            if gold_to_binary(gl, positive_from=positive_from) == 1:
                hits += 1

        per_q.append(hits / float(k))

    return sum(per_q) / float(len(per_q))


def build_grouped_inputs(
    eval_rows: List[Dict[str, object]],
    score_by_key: Dict[Tuple[str, str], float],
) -> Tuple[List[float], List[int], Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, int]]]:
    """
    Convenience helper.

    eval_rows: each row has {"query": str, "doc_id": str, "gold_label": int}
    score_by_key: {(query, doc_id): relevance_score}

    Returns:
    - flat_scores, flat_gold_binary (positive_from=1 is NOT applied here)
    - results_by_query for P@K: q -> [(doc_id, score)]
    - gold_by_query: q -> {doc_id -> gold_label}
    """
    flat_scores: List[float] = []
    flat_gold: List[int] = []

    results_by_query: Dict[str, List[Tuple[str, float]]] = {}
    gold_by_query: Dict[str, Dict[str, int]] = {}

    for row in eval_rows:
        q = str(row["query"])
        doc_id = str(row["doc_id"])
        gold_label = int(row["gold_label"])

        s = score_by_key.get((q, doc_id))
        if s is None:
            # skip rows without a score (e.g., doc missing / judgement failed)
            continue

        flat_scores.append(float(s))
        flat_gold.append(gold_label)

        results_by_query.setdefault(q, []).append((doc_id, float(s)))
        gold_by_query.setdefault(q, {})[doc_id] = gold_label

    return flat_scores, flat_gold, results_by_query, gold_by_query
