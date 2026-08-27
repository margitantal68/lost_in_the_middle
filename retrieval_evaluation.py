# retrieval_evaluation.py

import numpy as np
import chromadb
import csv

from typing import List, Tuple, Optional, Dict, Any, Callable
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_embeddings_to_arrays

def get_mrr_rank(retrieved_ids, gt_context):
    if isinstance(gt_context, str):
        gt_elements = [x.strip() for x in gt_context.split(',') if x.strip()]
    else:
        gt_elements = gt_context
    gt_set = {int(x) for x in gt_elements}
    for rank, r_id in enumerate(retrieved_ids, 1):
        if int(r_id) in gt_set:
            return rank
            
    return None



def evaluate_retrieval(
    question_embeddings: np.ndarray,
    question_context_ids: List[str],
    context_collection,
    question_ids: Optional[List[int]] = None,
    top_k: int = 5,
    output_csv: str = None,
) -> Dict[str, float]:
    """
    Evaluate retrieval metrics and save retrieval results to CSV.

    CSV format (no header):
    question_id, context_id,
    retrieved_id_1, retrieved_id_2, retrieved_id_3,
    retrieved_id_4, retrieved_id_5
    """

    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0

    print(f"Evaluating retrieval on {len(question_embeddings)} questions...")

    csv_file = None
    csv_writer = None

    if output_csv is not None:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

    for i, q_embed in enumerate(tqdm(question_embeddings, desc="Eval")):
        # Extract the question id from the proper dictionary
        question_id = question_ids[i]
        gt_context = str(question_context_ids[i])

        # Initial retrieval
        results = context_collection.query(
            query_embeddings=[q_embed.tolist() if hasattr(q_embed, "tolist") else q_embed],
            n_results=top_k
        )
        retrieved_ids = results.get("ids", [[]])[0]

        # Ensure exactly 5 retrieved IDs for CSV
        retrieved_ids_5 = retrieved_ids[:5] + [""] * (5 - len(retrieved_ids))

        # Write CSV row
        if csv_writer is not None:
            csv_writer.writerow(
                [question_id, gt_context] + retrieved_ids_5
            )

       
        rank = get_mrr_rank(retrieved_ids, gt_context)
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                recall_at_1 += 1
            if rank <= 3:
                recall_at_3 += 1
            if rank <= 5:
                recall_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    if csv_file is not None:
        csv_file.close()

    num_q = len(question_embeddings)
    mrr = float(np.mean(reciprocal_ranks))
    r1 = recall_at_1 / num_q
    r3 = recall_at_3 / num_q
    r5 = recall_at_5 / num_q

    print(f"MRR: {mrr:.4f}, Recall@1: {r1:.4f}, Recall@3: {r3:.4f}, Recall@5: {r5:.4f}")

    return {"MRR": mrr, "Recall@1": r1, "Recall@3": r3, "Recall@5": r5}


def evaluate_retrieval_2contexts(
    question_embeddings: np.ndarray,
    question_context_ids: List[str],
    context_collection,
    question_ids: Optional[List[int]] = None,
    top_k: int = 5,
    output_csv: str = None,
) -> Dict[str, float]:
    """
    Evaluate retrieval metrics for questions that have two ground-truth contexts.

    CSV format (no header):
    question_id, context_id,
    retrieved_id_1, retrieved_id_2, retrieved_id_3,
    retrieved_id_4, retrieved_id_5

    Scoring per question for Recall@k:
    - both GT contexts in top_k => 1.0
    - one GT context in top_k => 0.5
    - else => 0.0
    """
    print(f"Evaluating retrieval 2-contexts on {len(question_embeddings)} questions...")

    recall_scores = {3: 0.0, 4: 0.0, 5: 0.0}

    csv_file = None
    csv_writer = None

    if output_csv is not None:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

    for i, q_embed in enumerate(tqdm(question_embeddings, desc="Eval 2contexts")):
        question_id = question_ids[i] if question_ids is not None else i

        gt_context_raw = str(question_context_ids[i])
        gt_set = {
            int(x.strip())
            for x in gt_context_raw.split(",")
            if str(x).strip() != ""
        }

        results = context_collection.query(
            query_embeddings=[q_embed.tolist() if hasattr(q_embed, "tolist") else q_embed],
            n_results=top_k,
        )
        retrieved_ids = results.get("ids", [[]])[0]

        # normalize retrieved ids to ints where possible
        retrieved_ids = [int(x) for x in retrieved_ids if str(x).strip() != ""]

        retrieved_ids_5 = retrieved_ids[:5] + [""] * (5 - len(retrieved_ids))

        if csv_writer is not None:
            csv_writer.writerow([question_id, gt_context_raw] + retrieved_ids_5)

        for k in (3, 4, 5):
            topk_ids = set(retrieved_ids[:k])
            intersection = gt_set.intersection(topk_ids)
            cnt = len(intersection)

            if cnt == 2:
                score = 1.0
            elif cnt == 1:
                score = 0.5
            else:
                score = 0.0

            recall_scores[k] += score

    if csv_file is not None:
        csv_file.close()

    num_q = len(question_embeddings)
    recall_metrics = {
        "Recall@3": recall_scores[3] / num_q,
        "Recall@4": recall_scores[4] / num_q,
        "Recall@5": recall_scores[5] / num_q,
    }

    print(
        f"Recall@3: {recall_metrics['Recall@3']:.4f}, "
        f"Recall@4: {recall_metrics['Recall@4']:.4f}, "
        f"Recall@5: {recall_metrics['Recall@5']:.4f}"
    )

    return recall_metrics


def evaluate_retrieval_with_crossencoder(
    question_embeddings: np.ndarray,
    question_context_ids: List[str],
    question_texts: List[str],
    context_collection,
    context_texts_dict: Dict[str, str],
    crossencoder_model,
    top_k: int = 5,
    output_csv: str = None,
    question_ids: Optional[List[Any]] = None,
) -> Dict[str, float]:
    """
    Evaluate retrieval with CrossEncoder reranking.

    The initial embedding retrieval provides a shortlist, then CrossEncoder reranking
    reorders those candidates before metrics are computed. The CSV output is written
    in the reranked order to reflect the actual evaluation order.
    """
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0

    print(f"Evaluating retrieval with CrossEncoder reranker on {len(question_embeddings)} questions...")

    csv_file = None
    csv_writer = None

    if output_csv is not None:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

    for i, q_embed in enumerate(tqdm(question_embeddings, desc="Eval")):
        question_id = question_ids[i] if question_ids is not None else i
        gt_context_raw = str(question_context_ids[i])
        q_text = question_texts[i]

        results = context_collection.query(
            query_embeddings=[q_embed.tolist() if hasattr(q_embed, "tolist") else q_embed],
            n_results=top_k,
        )
        retrieved_ids = [str(s).strip() for s in results.get("ids", [[]])[0] if str(s).strip()]

        if not retrieved_ids:
            reciprocal_ranks.append(0.0)
            if csv_writer is not None:
                csv_writer.writerow([question_id, gt_context_raw] + [""] * 5)
            continue

        candidate_texts = []
        for cid in retrieved_ids:
            context_text = context_texts_dict.get(cid)
            if context_text is None:
                context_text = context_texts_dict.get(str(int(cid))) if str(cid).lstrip("-").isdigit() else None
            if context_text is None:
                continue
            candidate_texts.append((cid, context_text))

        if not candidate_texts:
            reciprocal_ranks.append(0.0)
            if csv_writer is not None:
                csv_writer.writerow([question_id, gt_context_raw] + [""] * 5)
            continue

        reranked_pairs = crossencoder_model.predict([(q_text, text) for _, text in candidate_texts])
        reranked_ids = [cid for _, cid in sorted(zip(reranked_pairs, [cid for cid, _ in candidate_texts]), key=lambda x: -float(x[0]))]

        gt_contexts = {str(x).strip() for x in gt_context_raw.split(",") if str(x).strip()}
        rank = None
        for rank, r_id in enumerate(reranked_ids, 1):
            if str(r_id).strip() in gt_contexts:
                break
        else:
            rank = None

        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                recall_at_1 += 1
            if rank <= 3:
                recall_at_3 += 1
            if rank <= 5:
                recall_at_5 += 1

        # keep the reranked order for the CSV and metric evaluation
        reranked_ids_5 = reranked_ids[:5] + [""] * (5 - len(reranked_ids))
        if csv_writer is not None:
            csv_writer.writerow([question_id, gt_context_raw] + reranked_ids_5)

    if csv_file is not None:
        csv_file.close()

    num_q = len(question_embeddings)
    mrr = float(np.mean(reciprocal_ranks))
    r1 = recall_at_1 / num_q
    r3 = recall_at_3 / num_q
    r5 = recall_at_5 / num_q
    print(f"MRR: {mrr:.4f}, Recall@1: {r1:.4f}, Recall@3: {r3:.4f}, Recall@5: {r5:.4f}")

    return {"MRR": mrr, "Recall@1": r1, "Recall@3": r3, "Recall@5": r5}



def compute_hubness_metrics(question_embeddings: np.ndarray, context_embeddings: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
    print(f"Computing hubness on {len(question_embeddings)} queries and {len(context_embeddings)} contexts...")

    sim_matrix = cosine_similarity(question_embeddings, context_embeddings)

    topk_indices = np.argpartition(-sim_matrix, kth=top_k, axis=1)[:, :top_k]
    neighbor_counts = Counter(topk_indices.flatten())

    Nk = np.zeros(len(context_embeddings))
    for idx, count in neighbor_counts.items():
        Nk[idx] = count

    Nk_mean = float(np.mean(Nk))
    Nk_std = float(np.std(Nk))
    Nk_skewness = float(skew(Nk))


    sorted_Nk = np.sort(Nk)
    n = len(sorted_Nk)
    Gini = (2 * np.sum((np.arange(1, n+1) * sorted_Nk)) / (n * np.sum(sorted_Nk))) - (n + 1) / n if np.sum(sorted_Nk) > 0 else 0.0

    most_common = neighbor_counts.most_common(10)

    print(f"Mean Nk: {Nk_mean:.2f}, Std: {Nk_std:.2f}, Skew: {Nk_skewness:.2f}, Gini: {Gini:.3f}")

    return {"Nk_mean": Nk_mean, "Nk_skewness": Nk_skewness, "Nk_std": Nk_std, "Gini": Gini, "Top10_hubs": most_common}


# if __name__ == "__main__":
#     # Tesztelés a példáddal:
#     retrieved_ids = [10, 20, 30, 40, 50]
#     # gt_context = "30, 50"
#     gt_context = "30"

#     rank = get_mrr_rank(retrieved_ids, gt_context)
#     print(f"A rank értéke: {rank}") # Eredmény: 3

