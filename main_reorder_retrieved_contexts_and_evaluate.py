# Write a script that does the following:
# Opens the data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv file, which contains the retrieved contexts for each question in the configured dataset
# in the following format: question_id, correct_context_id, context_id_1, context_id_2, ..., context_id_5
# Each question has one correct context and five retrieved contexts, which may or may not include the correct context.
# If the correct context is not in the retrieved contexts, leave the order of the retrieved contexts unchanged.
# If the correct context is in the retrieved contexts, swap the correct context with the first context in the list, so that the correct context is always the first one.
# Save the reordered contexts to a new file called data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_reordered_first.csv in the same format as the input file.

# Repeat the reordering again for the same input file, but this time swap the correct context with the last context in the list, so that the correct context is always the last one.
# Save the reordered contexts to a new file called data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_reordered_last.csv in the same format as the input file

# Repeat the reordering again for the same input file, but this time swap the correct context with the third context in the list, so that the correct context is always the third one.
# Save the reordered contexts to a new file called data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_reordered_middle.csv in the same format as the input file


import csv
from pathlib import Path
from typing import Dict, List

from config import DATASET, EMBEDDER_MODEL_NAME, QUESTIONS_PART, RETRIEVAL_TYPE


def resolve_input_path(model_name: str) -> Path:
    base_dir = Path("data/retriever")
    file_stem = f"{DATASET}_{RETRIEVAL_TYPE}_{model_name}"
    candidates = [
        base_dir / f"{file_stem}.csv",
        base_dir / f"{file_stem}.txt",
    ]

    if QUESTIONS_PART:
        candidates = [
            base_dir / f"{file_stem}_{QUESTIONS_PART}.csv",
            base_dir / f"{file_stem}_{QUESTIONS_PART}.txt",
            *candidates,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find an input file for dataset '{DATASET}', model '{model_name}' in {base_dir}"
    )


def reorder_row(row: list[str], target_position: int) -> list[str]:
    question_id = row[0]
    correct_context_id = row[1]
    retrieved_contexts = row[2:]

    if not retrieved_contexts:
        return row

    if correct_context_id not in retrieved_contexts:
        return row

    current_position = retrieved_contexts.index(correct_context_id)
    if current_position == target_position:
        return row

    reordered_contexts = retrieved_contexts.copy()
    reordered_contexts[current_position], reordered_contexts[target_position] = (
        reordered_contexts[target_position],
        reordered_contexts[current_position],
    )

    return [question_id, correct_context_id, *reordered_contexts]


def compute_retrieval_metrics(csv_path: Path) -> Dict[str, float]:
    num_queries = 0
    reciprocal_rank_sum = 0.0
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0

    with csv_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) < 7:
                continue

            num_queries += 1
            gold_context = row[1]
            retrieved_contexts = row[2:7]

            reciprocal_rank = 0.0
            for rank, context_id in enumerate(retrieved_contexts, start=1):
                if context_id == gold_context:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_rank_sum += reciprocal_rank

            if gold_context == retrieved_contexts[0]:
                recall_at_1 += 1
            if gold_context in retrieved_contexts[:3]:
                recall_at_3 += 1
            if gold_context in retrieved_contexts[:5]:
                recall_at_5 += 1

    if num_queries == 0:
        raise ValueError(f"No valid rows found in {csv_path}")

    return {
        "mrr": reciprocal_rank_sum / num_queries,
        "recall_at_1": recall_at_1 / num_queries,
        "recall_at_3": recall_at_3 / num_queries,
        "recall_at_5": recall_at_5 / num_queries,
    }


def collect_incorrect_retrieval_question_ids(csv_path: Path) -> List[str]:
    incorrect_question_ids: List[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) < 7:
                continue

            question_id = row[0]
            gold_context = row[1]
            retrieved_contexts = row[2:7]

            if gold_context not in retrieved_contexts:
                incorrect_question_ids.append(question_id)

    return incorrect_question_ids


def write_incorrect_retrieval_question_ids(question_ids: List[str], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        for question_id in question_ids:
            writer.writerow([question_id])


def reorder_file(input_path: Path, output_path: Path, target_position: int) -> int:
    with input_path.open("r", newline="") as infile:
        rows = [row for row in csv.reader(infile)]

    reordered_rows = [reorder_row(row, target_position) for row in rows]

    with output_path.open("w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(reordered_rows)

    missing_correct_context_count = sum(
        1 for row in reordered_rows if len(row) > 2 and row[1] not in row[2:]
    )
    return missing_correct_context_count


def main() -> None:
    input_path = resolve_input_path(EMBEDDER_MODEL_NAME)
    output_dir = input_path.parent

    input_stem = input_path.stem
    incorrect_ids_output_path = output_dir / (
        f"{input_stem}_incorrect_retrieval.csv"
    )
    incorrect_question_ids = collect_incorrect_retrieval_question_ids(input_path)
    write_incorrect_retrieval_question_ids(incorrect_question_ids, incorrect_ids_output_path)
    print(f"Wrote incorrect retrieval question IDs to {incorrect_ids_output_path}")
    print(f"Number of incorrect retrievals: {len(incorrect_question_ids)}")

    suffixes = {
        0: "reordered_first",
        4: "reordered_last",
        2: "reordered_middle",
    }

    for target_position, suffix in suffixes.items():
        output_path = output_dir / f"{input_path.stem}_{suffix}{input_path.suffix}"
        missing_count = reorder_file(input_path, output_path, target_position)
        print(f"Wrote {output_path}")
        print(
            f"Questions where correct context is not in retrieved contexts: {missing_count}"
        )

        metrics = compute_retrieval_metrics(output_path)
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Recall@1: {metrics['recall_at_1']:.4f}")
        print(f"Recall@3: {metrics['recall_at_3']:.4f}")
        print(f"Recall@5: {metrics['recall_at_5']:.4f}")


if __name__ == "__main__":
    main()

