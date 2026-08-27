import csv


def compute_retrieval_metrics(csv_filename):
    """
    Computes MRR, Recall@1, Recall@3, Recall@5 from a retriever CSV file
    WITHOUT a header line.

    Expected column order:
    0: question_id
    1: context_id (gold)
    2: retrieved_context_id1
    3: retrieved_context_id2
    4: retrieved_context_id3
    5: retrieved_context_id4
    6: retrieved_context_id5
    """

    num_queries = 0
    reciprocal_rank_sum = 0.0
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0

    with open(csv_filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            # Skip malformed rows if any
            if len(row) < 7:
                continue

            num_queries += 1

            gold_context = row[1]
            retrieved = row[2:7]

            # MRR
            rr = 0.0
            for rank, ctx in enumerate(retrieved, start=1):
                if ctx == gold_context:
                    rr = 1.0 / rank
                    break
            reciprocal_rank_sum += rr

            # Recall@k
            if gold_context == retrieved[0]:
                recall_at_1 += 1
            if gold_context in retrieved[:3]:
                recall_at_3 += 1
            if gold_context in retrieved[:5]:
                recall_at_5 += 1

    if num_queries == 0:
        print("No valid rows found in CSV.")
        return

    # Final metrics
    mrr = reciprocal_rank_sum / num_queries
    r1 = recall_at_1 / num_queries
    r3 = recall_at_3 / num_queries
    r5 = recall_at_5 / num_queries

    print(f"MRR: {mrr:.4f}")
    print(f"Recall@1: {r1:.4f}")
    print(f"Recall@3: {r3:.4f}")
    print(f"Recall@5: {r5:.4f}")


FOLDER = "data/retriever/"
# FILENAME = "top_k_retrieval"
FILENAME = "hubness_aware_reranking"

MODELS = ["BGE-BASE", "BGE-SMALL", "BGE-LARGE", "BGE-M3", "E5-BASE", "PP-MINILM", "XLMRoBERTaML"]

for i in range(len(MODELS)):
    print(f"\n--- Metrics for model: {MODELS[i]} ---")
    compute_retrieval_metrics(FOLDER + FILENAME + "_" + MODELS[i] + ".csv")