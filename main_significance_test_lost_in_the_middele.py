import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA_FOLDER = "data/answers/"

GEMMA_NONE   = "evaluated_output_answers_top_k_retrieval_E5-BASE_gemma.csv"
GEMMA_FIRST  = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_first_gemma.csv"
GEMMA_MIDDLE = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_middle_gemma.csv"
GEMMA_LAST   = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_gemma.csv"

LLAMA_NONE   = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_llama.csv"
LLAMA_FIRST  = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_first_llama.csv"
LLAMA_MIDDLE = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_middle_llama.csv"
LLAMA_LAST   = "_evaluated_output_answers_squad_top_k_retrieval_E5-BASE_reordered_last_llama.csv"


# LIVERAG_NONE   = "evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_newprompt.csv"
# LIVERAG_FIRST  = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_first_llama.csv"
# LIVERAG_MIDDLE = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_middle_llama.csv"
# LIVERAG_LAST   = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_last_llama.csv"

LIVERAG_NONE   = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_mistral.csv"
LIVERAG_FIRST  = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_first_mistral.csv"
LIVERAG_MIDDLE = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_middle_mistral.csv"
LIVERAG_LAST   = "_evaluated_output_answers_liverag_top_k_retrieval_BGE-M3_reordered_last_mistral.csv"


def _load_scores(data_folder: str, file_name: str) -> pd.DataFrame:
    path = os.path.join(data_folder, file_name)
    df = pd.read_csv(path)

    required = {"question_id", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {sorted(missing)}")

    out = df[["question_id", "score"]].copy()
    out["question_id"] = out["question_id"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["question_id", "score"]).drop_duplicates(subset="question_id")

    return out


def _paired_wilcoxon(df_a: pd.DataFrame, df_b: pd.DataFrame):
    merged = df_a.merge(df_b, on="question_id", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("No overlapping question_id values between the two measurement sets.")

    differences = merged["score_a"].to_numpy() - merged["score_b"].to_numpy()

    if np.allclose(differences, 0):
        return {
            "n": len(merged),
            "statistic": 0.0,
            "pvalue": 1.0,
        }

    result = wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return {
        "n": len(merged),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def compute_significance_tests(
    data_folder: str = DATA_FOLDER,
    alpha: float = 0.05,
    output_file: str = "wilcoxon_significance_results.csv",
    summary_file: str = "wilcoxon_significance_summary.csv",
):
    models = {
        # "gemma": {
        #     "NONE": GEMMA_NONE,
        #     "FIRST": GEMMA_FIRST,
        #     "MIDDLE": GEMMA_MIDDLE,
        #     "LAST": GEMMA_LAST,
        # },
        # "llama": {
        #     "NONE": LLAMA_NONE,
        #     "FIRST": LLAMA_FIRST,
        #     "MIDDLE": LLAMA_MIDDLE,
        #     "LAST": LLAMA_LAST,
        # },
        "liverag": {
            "NONE": LIVERAG_NONE,
            "FIRST": LIVERAG_FIRST,
            "MIDDLE": LIVERAG_MIDDLE,
            "LAST": LIVERAG_LAST,
        },
    }

    rows = []
    for model_name, files in models.items():
        loaded = {
            label: _load_scores(data_folder, file_name)
            for label, file_name in files.items()
        }

        for left, right in combinations(["NONE", "FIRST", "MIDDLE", "LAST"], 2):
            result = _paired_wilcoxon(loaded[left], loaded[right])
            significant = result["pvalue"] < alpha

            row = {
                "model": model_name,
                "left": left,
                "right": right,
                "n": result["n"],
                "statistic": result["statistic"],
                "pvalue": result["pvalue"],
                "alpha": alpha,
                "significant": bool(significant),
            }
            rows.append(row)

            print(
                f"{model_name}: {left} vs {right} | "
                f"n={row['n']} | statistic={row['statistic']:.6f} | "
                f"pvalue={row['pvalue']:.6e} | significant={row['significant']}"
            )

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(data_folder, output_file)
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby("model", as_index=False)
        .agg(
            total_pairs=("left", "size"),
            significant_pairs=("significant", lambda s: int(s.sum())),
            non_significant_pairs=("significant", lambda s: int((~s).sum())),
            min_pvalue=("pvalue", "min"),
            max_pvalue=("pvalue", "max"),
        )
    )
    summary_df["alpha"] = alpha
    summary_df["threshold"] = "pvalue < alpha"
    summary_df = summary_df[["model", "alpha", "threshold", "total_pairs", "significant_pairs", "non_significant_pairs", "min_pvalue", "max_pvalue"]]

    summary_path = os.path.join(data_folder, summary_file)
    summary_df.to_csv(summary_path, index=False)

    print("\nSignificance threshold summary:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved full results to: {results_path}")
    print(f"Saved summary to: {summary_path}")

    return results_df, summary_df


if __name__ == "__main__":
    compute_significance_tests()