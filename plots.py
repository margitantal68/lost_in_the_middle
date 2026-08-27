import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

OUTPUT_PATH = "plots/"


def plot_hubness_distribution(data_file_path):
    """
    Plot the ranked distribution of hubness scores, excluding rows where the
    correct supporting document is not present in the top-5 retrieved docs.

    Hubness score H(d) is defined as the number of times a document appears 
    among the top-k=5 retrieved documents across all queries, restricted to
    queries where the ground truth doc is actually retrieved.
    """
    df = pd.read_csv(data_file_path, header=None)
    valid_docs = []

    for _, row in df.iterrows():
        correct_doc = row[1]
        retrieved = list(row[2:7])
        if correct_doc in retrieved:
            valid_docs.extend(retrieved)

    if not valid_docs:
        print("No rows with the correct supporting document in the top-5 retrieved results.")
        return Counter()

    hubness_scores = Counter(valid_docs)
    sorted_hubness = sorted(hubness_scores.items(), key=lambda x: x[1], reverse=True)
    hubness_values = [score for _, score in sorted_hubness]

    fig, ax = plt.subplots(figsize=(12, 6))
    ranks = range(1, len(hubness_values) + 1)
    ax.plot(ranks, hubness_values, marker='o', linewidth=1.5, markersize=4)
    ax.fill_between(ranks, hubness_values, alpha=0.3)

    ax.set_xlabel('Document Rank (sorted by hubness score)', fontsize=12)
    ax.set_ylabel('Hubness Score H(d)', fontsize=12)
    ax.set_title('Ranked Distribution of Hubness Scores (Top-5 Retrieved Documents, Correct Doc Present)', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_path = f"{OUTPUT_PATH}{data_file_path.split('/')[-1].replace('.csv', '_hubness_distribution.png')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()

    print(f"\nFiltered Hubness Statistics:")
    print(f"Total unique documents in retrieval results: {len(hubness_scores)}")
    print(f"Total retrievals included: {len(valid_docs)}")
    print(f"Max hubness score: {max(hubness_values)}")
    print(f"Min hubness score: {min(hubness_values)}")
    print(f"Mean hubness score: {sum(hubness_values) / len(hubness_values):.2f}")

    return hubness_scores




def plot_precedence_hubness_distribution(data_file_path):
    """
    Plot the ranked distribution of precedence hubness scores, excluding rows
    where the correct supporting document does not appear in the top-5 retrieved docs.
    """
    df = pd.read_csv(data_file_path, header=None)
    precedence_counts = Counter()
    included_rows = 0

    for _, row in df.iterrows():
        correct_doc = row[1]
        retrieved = list(row[2:7])

        if correct_doc not in retrieved:
            continue

        correct_index = retrieved.index(correct_doc)
        ahead_docs = retrieved[:correct_index]
        precedence_counts.update(ahead_docs)
        included_rows += 1

    if not precedence_counts:
        print("No rows with the correct supporting document in the top-5 retrieved results.")
        return Counter()

    sorted_precedence = sorted(precedence_counts.items(), key=lambda x: x[1], reverse=True)
    precedence_values = [score for _, score in sorted_precedence]

    fig, ax = plt.subplots(figsize=(12, 6))
    ranks = range(1, len(precedence_values) + 1)
    ax.plot(ranks, precedence_values, marker='o', linewidth=1.5, markersize=4)
    ax.fill_between(ranks, precedence_values, alpha=0.3)

    ax.set_xlabel('Document Rank (sorted by precedence hubness score)', fontsize=12)
    ax.set_ylabel('Precedence Hubness Score P(d)', fontsize=12)
    ax.set_title('Ranked Distribution of Precedence Hubness Scores (Top-5 Retrieved Documents, Correct Doc Present)', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_path = f"{OUTPUT_PATH}{data_file_path.split('/')[-1].replace('.csv', '_precedence_hubness_distribution.png')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()

    print(f"\nFiltered Precedence Hubness Statistics:")
    print(f"Total unique documents considered: {len(precedence_counts)}")
    print(f"Total included queries: {included_rows}")
    print(f"Total precedence events counted: {sum(precedence_counts.values())}")
    print(f"Max precedence hubness score: {max(precedence_values)}")
    print(f"Min precedence hubness score: {min(precedence_values)}")
    print(f"Mean precedence hubness score: {sum(precedence_values) / len(precedence_values):.2f}")

    return precedence_counts


if __name__ == "__main__":
    # Call the function for the SQuAD E5-BASE retrieval results
    # plot_hubness_distribution("data/retriever/liverag_top_k_retrieval_BGE-M3.csv")
    # plot_precedence_hubness_distribution("data/retriever/liverag_top_k_retrieval_BGE-M3.csv")

    # plot_hubness_distribution("data/retriever/squad_top_k_retrieval_E5-BASE.csv")
    # plot_precedence_hubness_distribution("data/retriever/squad_top_k_retrieval_E5-BASE.csv")

    # plot_hubness_distribution("data/retriever/squad_top_k_retrieval_BGE-M3.csv")
    # plot_precedence_hubness_distribution("data/retriever/squad_top_k_retrieval_BGE-M3.csv")

    plot_hubness_distribution("data/retriever/squad_top_k_retrieval_BGE-BASE.csv")
    plot_precedence_hubness_distribution("data/retriever/squad_top_k_retrieval_BGE-BASE.csv")


