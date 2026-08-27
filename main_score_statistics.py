import csv
import statistics
from collections import Counter
from typing import Dict, List, Set
from config import DATASET, EMBEDDER_MODEL_NAME, QUESTIONS_PART, RETRIEVAL_TYPE

# QUESTIONID_FILE = f"data/retriever/{DATASET}_top_k_retrieval_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_incorrect_retrieval.csv"

# LIVERAG
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_last_mistral.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_mistral.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_newprompt.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_first_gemma.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_middle_gemma.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_gemma.csv"

# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_first_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_middle_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_llama.csv"


# SQUAD
# QUESTIONID_FILE = f"data/retriever/{DATASET}_top_k_retrieval_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_incorrect_retrieval.csv"
QUESTIONID_FILE = f"data/retriever/squad_top_k_retrieval_E5-BASE_incorrect_retrieval.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_crossencoder_E5-BASE_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_liverag_crossencoder_BGE-M3_mistral.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_mistral.csv"
EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_crossencoder_E5-BASE_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_top_k_retrieval_E5-BASE_gemma.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_mistral.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_middle_gemma.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_gemma.csv"

# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_first_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_middle_llama.csv"
# EVALUATED_ANSWER_FILE = "data/answers/_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_llama.csv"

# GEMMA
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_0_fold_0.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_0_fold_1.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_1_fold_0.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_1_fold_1.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_271_fold_0.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_271_fold_1.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_314_fold_0.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_314_fold_1.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_42_fold_0.csv"
# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_42_fold_1.csv"

# EVALUATED_ANSWER_FILE = "data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_1_fold_1.csv"

def load_question_ids(question_id_file: str) -> Set[str]:
    question_ids: Set[str] = set()

    with open(question_id_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue
            question_id = row[0].strip()
            if question_id and question_id != "question_id":
                question_ids.add(question_id)

    return question_ids


def compute_score_statistics(question_id_file: str, evaluated_answer_file: str) -> Dict[str, float]:
    question_ids = load_question_ids(question_id_file)
    score_values: List[float] = []
    score_counter: Counter = Counter()

    with open(evaluated_answer_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue

            if not row[0].strip():
                continue

            if row[0].strip().lower() == "question_id":
                continue

            if len(row) < 6:
                continue

            question_id = row[0].strip()
            if question_id in question_ids:
                try:
                    score = float(row[4].strip())
                    score_values.append(score)
                    score_counter[int(score)] += 1
                except ValueError:
                    continue

    if not score_values:
        raise ValueError(
            f"No matching scores found for question IDs from {question_id_file} in {evaluated_answer_file}"
        )

    return {
        "count": float(len(score_values)),
        "mean": float(statistics.mean(score_values)),
        "median": float(statistics.median(score_values)),
        "min": float(min(score_values)),
        "max": float(max(score_values)),
        "std": float(statistics.pstdev(score_values)),
        "q1": float(statistics.quantiles(score_values, n=4)[0]),
        "q3": float(statistics.quantiles(score_values, n=4)[2]),
        "distribution": {str(score): score_counter.get(score, 0) for score in range(1, 6)},
    }


def compute_score_statistics_for_non_question_ids(question_id_file: str, evaluated_answer_file: str) -> Dict[str, float]:
    question_ids = load_question_ids(question_id_file)
    score_values: List[float] = []
    score_counter: Counter = Counter()

    with open(evaluated_answer_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue

            if not row[0].strip():
                continue

            if row[0].strip().lower() == "question_id":
                continue

            if len(row) < 6:
                continue

            question_id = row[0].strip()
            if question_id not in question_ids:
                try:
                    score = float(row[4].strip())
                    score_values.append(score)
                    score_counter[int(score)] += 1
                except ValueError:
                    continue

    if not score_values:
        raise ValueError(
            f"No matching scores found for questions not in {question_id_file} in {evaluated_answer_file}"
        )

    return {
        "count": float(len(score_values)),
        "mean": float(statistics.mean(score_values)),
        "median": float(statistics.median(score_values)),
        "min": float(min(score_values)),
        "max": float(max(score_values)),
        "std": float(statistics.pstdev(score_values)),
        "q1": float(statistics.quantiles(score_values, n=4)[0]),
        "q3": float(statistics.quantiles(score_values, n=4)[2]),
        "distribution": {str(score): score_counter.get(score, 0) for score in range(1, 6)},
    }

def compute_all_score_statistics(evaluated_answer_file: str) -> Dict[str, float]:
    score_values: List[float] = []
    score_counter: Counter = Counter()

    with open(evaluated_answer_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue

            if not row[0].strip():
                continue

            if row[0].strip().lower() == "question_id":
                continue

            if len(row) < 6:
                continue

            try:
                score = float(row[4].strip())
                score_values.append(score)
                score_counter[int(score)] += 1
            except ValueError:
                continue

    if not score_values:
        raise ValueError(f"No valid scores found in {evaluated_answer_file}")

    return {
        "count": float(len(score_values)),
        "mean": float(statistics.mean(score_values)),
        "median": float(statistics.median(score_values)),
        "min": float(min(score_values)),
        "max": float(max(score_values)),
        "std": float(statistics.pstdev(score_values)),
        "q1": float(statistics.quantiles(score_values, n=4)[0]),
        "q3": float(statistics.quantiles(score_values, n=4)[2]),
        "distribution": {str(score): score_counter.get(score, 0) for score in range(1, 6)},
    }


def print_statistics(title: str, stats: Dict[str, object]) -> None:
    print(title)
    for key, value in stats.items():
        if key == "distribution":
            print("distribution:")
            for score, count in value.items():
                print(f"  {score}: {count}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print()


if __name__ == "__main__":
    stats = compute_score_statistics(QUESTIONID_FILE, EVALUATED_ANSWER_FILE)
    print_statistics("Score statistics for questions in ", stats)

    other_stats = compute_score_statistics_for_non_question_ids(QUESTIONID_FILE, EVALUATED_ANSWER_FILE)
    print_statistics("Score statistics for questions not in ", other_stats)

    all_stats = compute_all_score_statistics(EVALUATED_ANSWER_FILE)
    print_statistics("Score statistics for all questions", all_stats)

    
    
