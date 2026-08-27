"""Exploratory Data Analysis for SQuAD v2.0 answerable Dev questions.

This script reads the answerable portion of the SQuAD v2.0 Dev split
from data/squad/dev-v2.0.json and produces summary statistics plus
visualizations for question, answer, and context length distributions.
"""

from pathlib import Path
from typing import Dict, List, Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import extract_squad_data

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

DEFAULT_SQUAD_PATH = Path("data/squad/dev-v2.0.json")
DEFAULT_PLOT_DIR = Path("plots/squad_eda")


def load_squad_answerable_dev(file_path: Path = DEFAULT_SQUAD_PATH) -> pd.DataFrame:
    """Load answerable SQuAD v2.0 Dev questions into a pandas DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"SQuAD file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        squad_data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for article_index, article in enumerate(squad_data.get("data", [])):
        title = article.get("title", "")
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            context_id = f"{article_index}-{paragraph_index}"

            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible", False):
                    continue

                answers = qa.get("answers", [])
                if not answers:
                    continue

                rows.append(
                    {
                        "question_id": qa.get("id", ""),
                        "article_title": title,
                        "paragraph_id": paragraph_index,
                        "context_id": context_id,
                        "context_text": context,
                        "question_text": qa.get("question", ""),
                        "answer_texts": [answer.get("text", "") for answer in answers],
                        "answer_starts": [answer.get("answer_start", -1) for answer in answers],
                        "answer_count": len(answers),
                        "first_answer_text": answers[0].get("text", ""),
                        "first_answer_start": answers[0].get("answer_start", -1),
                    }
                )

    if not rows:
        raise ValueError("No answerable questions were found in the SQuAD Dev file.")

    df = pd.DataFrame(rows)
    df["context_char_len"] = df["context_text"].str.len()
    df["context_word_count"] = df["context_text"].str.split().str.len()
    df["question_char_len"] = df["question_text"].str.len()
    df["question_word_count"] = df["question_text"].str.split().str.len()
    df["answer_char_len"] = df["first_answer_text"].str.len()
    df["answer_word_count"] = df["first_answer_text"].str.split().str.len()
    df["max_answer_char_len"] = df["answer_texts"].apply(lambda texts: max((len(text) for text in texts), default=0))
    df["max_answer_word_count"] = df["answer_texts"].apply(lambda texts: max((len(text.split()) for text in texts), default=0))
    df["answer_start"] = df["first_answer_start"]

    # Validate using the helper from utils.py
    contexts_dict, questions_dict = extract_squad_data(str(file_path))
    if len(df) != len(questions_dict):
        print("WARNING: Extracted question count differs from utils.extract_squad_data().")
        print(f"  DataFrame rows: {len(df)}")
        print(f"  utils.extract_squad_data question count: {len(questions_dict)}")

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print high-level dataset statistics."""
    total_questions = len(df)
    total_contexts = df["context_id"].nunique()
    total_articles = df["article_title"].nunique()
    total_paragraphs = df[["article_title", "paragraph_id"]].drop_duplicates().shape[0]

    question_counts_per_context = df["context_id"].value_counts()
    answer_counts_per_question = df["answer_count"].value_counts().sort_index()

    print("\n" + "=" * 80)
    print("SQuAD v2.0 Dev Answerable Question EDA")
    print("=" * 80)
    print(f"Total answerable questions: {total_questions}")
    print(f"Total unique contexts: {total_contexts}")
    print(f"Total articles: {total_articles}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Average questions per context: {question_counts_per_context.mean():.2f}")
    print(f"Median questions per context: {question_counts_per_context.median():.2f}")
    print(f"Max questions in a single context: {question_counts_per_context.max()}")
    print(f"Questions with multiple answer annotations: {len(df[df['answer_count'] > 1])}")

    print("\nTop 10 contexts by question count:")
    for context_id, count in question_counts_per_context.head(10).items():
        context_preview = df.loc[df["context_id"] == context_id, "context_text"].iloc[0][:120].replace("\n", " ")
        print(f"  {context_id}: {count} questions  |  {context_preview}...")

    print("\nAnswer count distribution per question:")
    for answer_count, count in answer_counts_per_question.items():
        print(f"  {answer_count:2d} answers : {count:6d} questions")

    print("\nBasic text length statistics:")
    print(f"  Question chars  : mean={df['question_char_len'].mean():.1f}, median={df['question_char_len'].median():.1f}, min={df['question_char_len'].min()}, max={df['question_char_len'].max()}")
    print(f"  Question words  : mean={df['question_word_count'].mean():.1f}, median={df['question_word_count'].median():.1f}, min={df['question_word_count'].min()}, max={df['question_word_count'].max()}")
    print(f"  Answer chars    : mean={df['answer_char_len'].mean():.1f}, median={df['answer_char_len'].median():.1f}, min={df['answer_char_len'].min()}, max={df['answer_char_len'].max()}")
    print(f"  Answer words    : mean={df['answer_word_count'].mean():.1f}, median={df['answer_word_count'].median():.1f}, min={df['answer_word_count'].min()}, max={df['answer_word_count'].max()}")
    print(f"  Context chars   : mean={df['context_char_len'].mean():.1f}, median={df['context_char_len'].median():.1f}, min={df['context_char_len'].min()}, max={df['context_char_len'].max()}")
    print(f"  Context words   : mean={df['context_word_count'].mean():.1f}, median={df['context_word_count'].median():.1f}, min={df['context_word_count'].min()}, max={df['context_word_count'].max()}")


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_length_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["question_char_len"], bins=40, kde=False, ax=axes[0, 0], color="#4c72b0")
    axes[0, 0].set_title("Question Length Distribution (chars)")
    axes[0, 0].set_xlabel("Characters")

    sns.histplot(df["question_word_count"], bins=40, kde=False, ax=axes[0, 1], color="#dd8452")
    axes[0, 1].set_title("Question Length Distribution (words)")
    axes[0, 1].set_xlabel("Words")

    sns.histplot(df["answer_char_len"], bins=40, kde=False, ax=axes[1, 0], color="#55a868")
    axes[1, 0].set_title("Answer Length Distribution (chars)")
    axes[1, 0].set_xlabel("Characters")

    sns.histplot(df["context_char_len"], bins=40, kde=False, ax=axes[1, 1], color="#c44e52")
    axes[1, 1].set_title("Context Length Distribution (chars)")
    axes[1, 1].set_xlabel("Characters")

    fig.suptitle("SQuAD v2.0 Dev Answerable Length Distributions", fontsize=16)
    save_plot(fig, output_dir / "squad_length_distributions.png")


def plot_answer_count_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_counts = df["answer_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=answer_counts.index.astype(str), y=answer_counts.values, palette="viridis", ax=ax)
    ax.set_title("Number of Answer Annotations per Question")
    ax.set_xlabel("Answer count")
    ax.set_ylabel("Number of questions")
    save_plot(fig, output_dir / "squad_answer_count_distribution.png")


def plot_context_question_usage(df: pd.DataFrame, output_dir: Path, top_k: int = 15) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_usage = df["context_id"].value_counts().head(top_k)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=context_usage.values, y=context_usage.index, palette="rocket", ax=ax)
    ax.set_title(f"Top {top_k} Contexts by Number of Questions")
    ax.set_xlabel("Number of questions")
    ax.set_ylabel("Context ID")
    save_plot(fig, output_dir / "squad_top_context_usage.png")


def plot_answer_start_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["answer_start"], bins=60, kde=False, color="#7b3294", ax=ax)
    ax.set_title("Answer Start Position Distribution")
    ax.set_xlabel("Answer start character index")
    ax.set_ylabel("Question count")
    save_plot(fig, output_dir / "squad_answer_start_distribution.png")


def run_eda(file_path: Path = DEFAULT_SQUAD_PATH, output_dir: Path = DEFAULT_PLOT_DIR) -> None:
    df = load_squad_answerable_dev(file_path)
    print_dataset_summary(df)
    plot_length_histograms(df, output_dir)
    plot_answer_count_distribution(df, output_dir)
    plot_context_question_usage(df, output_dir)
    plot_answer_start_distribution(df, output_dir)
    print(f"\nSaved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    run_eda()
