import csv

from config import DATASET, RETRIEVAL_TYPE, EMBEDDER_MODEL_NAME


def _normalize_context_ids(value):
    if value is None:
        return []

    if isinstance(value, str):
        raw_values = value.replace(";", ",").split(",")
    else:
        raw_values = [value]

    normalized = []
    for item in raw_values:
        item = str(item).strip()
        if item:
            normalized.append(item)
    return normalized


def compute_retrieval_metrics(csv_path):
    """Compute MRR and Recall@k for a retrieval CSV.

    Expected CSV columns:
    questionid,gold_context_id,retrieved_context_id_1,...,retrieved_context_id_5
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return {
            "MRR": 0.0,
            "Recall@1": 0.0,
            "Recall@2": 0.0,
            "Recall@3": 0.0,
            "Recall@4": 0.0,
            "Recall@5": 0.0,
        }

    header = [str(cell).strip().lower() for cell in raw_rows[0]]
    if "questionid" in header or "gold_context_id" in header:
        rows = raw_rows[1:]
    else:
        rows = raw_rows

    reciprocal_rank_sum = 0.0
    recall_counts = {k: 0 for k in (1, 2, 3, 4, 5)}
    num_queries = 0

    for row in rows:
        if len(row) < 7:
            continue

        gold_context_ids = _normalize_context_ids(row[1])
        if not gold_context_ids:
            continue

        retrieved_ids = [str(item).strip() for item in row[2:7] if str(item).strip()]
        if not retrieved_ids:
            continue

        num_queries += 1

        first_hit_rank = None
        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in gold_context_ids:
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            reciprocal_rank_sum += 1.0 / first_hit_rank

        for k in (1, 2, 3, 4, 5):
            topk = set(retrieved_ids[:k])
            if any(gold_id in topk for gold_id in gold_context_ids):
                recall_counts[k] += 1

    if num_queries == 0:
        metrics = {"MRR": 0.0, "Recall@1": 0.0, "Recall@2": 0.0, "Recall@3": 0.0, "Recall@4": 0.0, "Recall@5": 0.0}
        return metrics

    metrics = {
        "MRR": reciprocal_rank_sum / num_queries,
        "Recall@1": recall_counts[1] / num_queries,
        "Recall@2": recall_counts[2] / num_queries,
        "Recall@3": recall_counts[3] / num_queries,
        "Recall@4": recall_counts[4] / num_queries,
        "Recall@5": recall_counts[5] / num_queries,
    }

    print(f"MRR: {metrics['MRR']:.4f}")
    for k in (1, 2, 3, 4, 5):
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.4f}")

    return metrics


if DATASET == "liverag":
    retrieval_file = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_1.csv"
else:
    retrieval_file = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv"

# CSV: questionid, gold_context_id, retrieved_context_id_1, retrieved_context_id_2, retrieved_context_id_3, retrieved_context_id_4, retrieved_context_id_5


if __name__ == "__main__":
    print(f"Computing retrieval metrics for {RETRIEVAL_TYPE} with {EMBEDDER_MODEL_NAME} on {DATASET} dataset...")
    print(f"Using retrieval CSV: {retrieval_file}")
    compute_retrieval_metrics(retrieval_file)
