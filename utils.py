import statistics
import glob
import os
import numpy as np
import json
import numpy as np
import chromadb
import csv
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Tuple, List, Any
from chromadb.utils import embedding_functions
from collections import defaultdict
from transformers import AutoTokenizer



def extract_squad_data(file_path: str) -> Tuple[Dict[int, str], Dict[int, Tuple[str, int]]]:
    """Parse SQuAD v2 JSON and return contexts and questions mapping.
    Returns:
    contexts_dict: {context_id: context_text}
    questions_dict: {question_id: (question_text, context_id)}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    contexts_dict: Dict[int, str] = {}
    questions_dict: Dict[int, Tuple[str, int]] = {}
    context_id = 0
    question_id = 0

    for article in squad_data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")


            # Store context (avoid duplicates)
            if context not in contexts_dict.values():
                contexts_dict[context_id] = context
                current_context_id = context_id
                context_id += 1
            else:
                current_context_id = [k for k, v in contexts_dict.items() if v == context][0]

            for qa in paragraph.get("qas", []):
                if not qa.get("is_impossible", False):
                    questions_dict[question_id] = (qa.get("question", ""), current_context_id)
                    question_id += 1

    return contexts_dict, questions_dict

# Export / import helpers for embeddings JSON format used in this project

def save_embeddings_json(items: List[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def load_embeddings_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_embeddings_to_arrays(items: List[dict]) -> Tuple[List[str], List[str], np.ndarray, List[Any]]:
    """Convert exported embedding JSON list into ids, texts, numpy array of embeddings, and optional metadata list.

    Expected item format per element: {"id": "123", "text": "...", "embedding": [..], "metadata": {...} }
    """
    ids = [item["id"] for item in items]
    texts = [item.get("text", "") for item in items]
    embeds = np.array([item["embedding"] for item in items], dtype=float)
    metadatas = [item.get("metadata") for item in items]
    return ids, texts, embeds, metadatas


def create_chroma_collection(
    collection_name: str,
    ids: List[str],
    texts: List[str],
    embeddings: np.ndarray,
    persist_dir: str = "chroma_db"
):
    """
    Create and populate a persistent Chroma collection.
    """
    client = chromadb.PersistentClient(path=persist_dir)

    # Remove existing collection if it already exists
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection didn't previously exist

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )

    collection.add(
        ids=[str(i) for i in ids],
        documents=texts,
        embeddings=embeddings.tolist(),
    )

    return collection


def load_chroma_collection(
    collection_name: str,
    persist_dir: str = "chroma_db"
):
    """
    Load an existing persistent Chroma collection.
    Returns None if not found.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        return client.get_collection(collection_name)
    except Exception:
        return None


def get_or_create_context_collection(
    collection_name: str,
    ids: List[str],
    texts: List[str],
    embeddings: np.ndarray,
    persist_dir: str = "chroma_db"
):
    """
    Load the Chroma context collection if it exists.
    Otherwise create it.
    """
    collection = load_chroma_collection(collection_name, persist_dir)

    if collection is not None:
        print(f"Loaded existing Chroma collection: {collection_name}")
        return collection

    print(f"Creating new Chroma collection: {collection_name}")
    return create_chroma_collection(
        collection_name,
        ids,
        texts,
        embeddings,
        persist_dir=persist_dir,
    )

def create_chroma_collection_from_embeddings(
items: List[dict], collection_name: str = "contexts_from_export"
) -> Tuple[Any, np.ndarray]:
    """Create or recreate a Chroma collection from exported items (which contain embeddings).
    Returns (chroma_collection, embeddings_array) where embeddings_array is numpy array in the same order as collection ids.
    """
    client = chromadb.Client()

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name)
    ids, texts, embeddings, metadatas = load_embeddings_to_arrays(items)

    # Chroma expects lists (embeddings as lists)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    return collection, embeddings

def export_questions_to_csv():
    contexts_dict, questions_dict = extract_squad_data("data/squad/dev-v2.0.json")
    OUTPUT_FILE = "data/answers/squad_questions.csv"
    # Export questions to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question_text"])
        for qid, (qtext, _) in questions_dict.items():
            writer.writerow([qid, qtext])  



def compute_score_distribution(csv_file_path: str, score_column: str = "score") -> None:
    """
    Compute and print the distribution of scores from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        score_column: Name of the column containing scores (default: "score")
    
    Scores are expected to be integers in range 1-5.
    Prints:
        - Distribution of scores (count and percentage for each score 1-5)
        - First order statistics (mean, median)
        - Second order statistics (variance, standard deviation)
    """
    scores = []
    
    # Read scores from CSV
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or score_column not in reader.fieldnames:
            print(f"Error: Column '{score_column}' not found in CSV file")
            return
        
        for row in reader:
            try:
                score = int(row[score_column])
                scores.append(score)
            except ValueError:
                print(f"Warning: Could not convert '{row[score_column]}' to integer, skipping")
                continue
    
    if not scores:
        print("No valid scores found in CSV file")
        return
    
    scores_array = np.array(scores)
    
    # Compute distribution (count for each score 1-5)
    distribution = {i: 0 for i in range(1, 6)}
    for score in scores:
        if 1 <= score <= 5:
            distribution[score] += 1
    
    # Print header
    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Print distribution
    print("\nScore Distribution:")
    print("-" * 60)
    total_scores = len(scores)
    for score in range(1, 6):
        count = distribution[score]
        percentage = (count / total_scores) * 100
        print(f"Score {score}: {count:4d} ({percentage:6.2f}%)")
    print("-" * 60)
    print(f"Total scores: {total_scores}")
    
    # Compute first order statistics (mean, median)
    mean_score = np.mean(scores_array)
    median_score = np.median(scores_array)
    
    print("\nFirst Order Statistics (Location):")
    print("-" * 60)
    print(f"Mean:   {mean_score:.4f}")
    print(f"Median: {median_score:.4f}")
    
    # Compute second order statistics (variance, standard deviation)
    variance = np.var(scores_array)
    std_dev = np.std(scores_array)
    
    print("\nSecond Order Statistics (Spread):")
    print("-" * 60)
    print(f"Variance:      {variance:.4f}")
    print(f"Std Deviation: {std_dev:.4f}")
    print("=" * 60 + "\n")




def squad_statistics():
    """
    Compute and display statistics about context token counts.
    """
    SQUAD_PATH = "data/squad/dev-v2.0.json"
    TOKENIZER_MODEL = "BAAI/bge-m3"
    # TOKENIZER_MODEL = "intfloat/multilingual-e5-base"

    # Initialize MAX_TOKENS with the sequence length of the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    MAX_TOKENS = tokenizer.model_max_length
    print(f"Using tokenizer from model: {TOKENIZER_MODEL}")
    print(f"Model's tokenizer max sequence length (MAX_TOKENS): {MAX_TOKENS}\n")    

    print(f"\nParsing SQuAD at: {SQUAD_PATH}")
    contexts, questions = extract_squad_data(SQUAD_PATH)

    print(f"Extracted {len(contexts)} unique contexts and {len(questions)} questions.\n")
    # Compute token counts for each context
    token_counts = []
    context_ids = []

    contexts_exceeding_MAX_TOKENS = []

    for context_id, context_text in contexts.items():
        token_count = len(tokenizer.encode(context_text))
        token_counts.append(token_count)
        context_ids.append(context_id)
        
        if token_count > MAX_TOKENS:
            contexts_exceeding_MAX_TOKENS.append((context_id, token_count))

    # Convert to numpy array for statistics computation
    token_counts_array = np.array(token_counts)

    # Compute statistics
    min_tokens = np.min(token_counts_array)
    max_tokens = np.max(token_counts_array)
    avg_tokens = np.mean(token_counts_array)
    std_tokens = np.std(token_counts_array)

    # Print statistics
    print("=" * 50)
    print("CONTEXT TOKEN COUNT STATISTICS")
    print("=" * 50)
    print(f"Number of contexts: {len(contexts)}")
    print(f"Minimum tokens:     {min_tokens}")
    print(f"Maximum tokens:     {max_tokens}")
    print(f"Average tokens:     {avg_tokens:.2f}")
    print(f"Std deviation:      {std_tokens:.2f}")
    print("=" * 50)

    # Print contexts exceeding MAX_TOKENS tokens
    print(f"\nContexts exceeding {MAX_TOKENS} tokens: {len(contexts_exceeding_MAX_TOKENS)}")
    print("=" * 50)
    for context_id, token_count in sorted(contexts_exceeding_MAX_TOKENS, key=lambda x: x[1], reverse=True):
        print(f"Context ID: {context_id}, Tokens: {token_count}")
    print("=" * 50)


def extract_squad_data_with_topics(file_path: str) -> Tuple[Dict[int, Tuple[str, str]], Dict[int, Tuple[str, int]]]:
    """Parse SQuAD v2 JSON and return contexts and questions mapping including topic/title.
    Returns:
    contexts_dict: {context_id: (context_text, context_topic)}
    questions_dict: {question_id: (question_text, context_id)}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    contexts_dict: Dict[int, Tuple[str, str]] = {}
    questions_dict: Dict[int, Tuple[str, int]] = {}
    context_id = 0
    question_id = 0

    for article in squad_data.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")

            # Store context with topic (avoid duplicates by context text)
            if context not in [v[0] for v in contexts_dict.values()]:
                contexts_dict[context_id] = (context, title)
                current_context_id = context_id
                context_id += 1
            else:
                current_context_id = [k for k, v in contexts_dict.items() if v[0] == context][0]

            for qa in paragraph.get("qas", []):
                if not qa.get("is_impossible", False):
                    questions_dict[question_id] = (qa.get("question", ""), current_context_id)
                    question_id += 1

    return contexts_dict, questions_dict


def export_squad_topics_csv(file_path: str = "data/squad/dev-v2.0.json", output_file: str = "data/answers/squad_topics.csv") -> None:
    """Export CSV with columns: question_id, question_text, context_id, context_topic, context_text

    The default input and output match the project's data layout.
    """
    contexts, questions = extract_squad_data_with_topics(file_path)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question_text", "context_id", "context_topic", "context_text"])
        for qid, (qtext, ctx_id) in questions.items():
            ctx_text, ctx_topic = contexts[ctx_id]
            writer.writerow([qid, qtext, ctx_id, ctx_topic, ctx_text])
# new ordered statistics function (keeps the existing commented version intact below)


def compute_topic_score_statistics(retrieval_type: str, embedding_model_name: str,
                                   top_k_topics: int = 30) -> dict:
    """Compute per-topic score statistics from evaluated answer CSV(s) and save plots.

    Args:
        evaluated_csv_paths: single path or glob pattern or list of paths to evaluated_output CSV files.
        squad_file: path to SQuAD JSON (used to map question_id -> context/topic).
        output_dir: directory where summary CSV and plots will be saved.
        top_k_topics: number of top topics by sample count to include in the boxplot.

    Returns:
        A dict mapping topic -> statistics dict (count, mean, median, std, var, distribution dict).
    """


    squad_file: str = "data/squad/dev-v2.0.json"
    evaluated_csv_paths = f"data/answers/evaluated_output_answers_{retrieval_type}_{embedding_model_name}.csv"
    output_dir = "data/answers/topics/"
    output_file = f"topic_stats_{retrieval_type}_{embedding_model_name}.csv"
 

    # Normalize input paths
    if isinstance(evaluated_csv_paths, str):
        # If a glob pattern, expand; otherwise treat as single path
        if any(ch in evaluated_csv_paths for ch in ["*", "?", "["]):
            paths = glob.glob(evaluated_csv_paths)
        else:
            paths = [evaluated_csv_paths]
    else:
        paths = list(evaluated_csv_paths)

    if not paths:
        raise ValueError("No evaluated CSV paths provided/found")

    # Load mapping from questions to topics
    contexts, questions = extract_squad_data_with_topics(squad_file)

    # Build mapping question_id -> topic
    qid_to_topic = {}
    qid_to_context = {}
    for qid, (_qtext, ctx_id) in questions.items():
        ctx_text, ctx_topic = contexts[ctx_id]
        qid_to_topic[int(qid)] = ctx_topic
        qid_to_context[int(qid)] = (ctx_id, ctx_text)

    # Aggregate scores per topic
    topic_scores = defaultdict(list)

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    qid = int(row.get("question_id", row.get("question_id")))
                except Exception:
                    continue
                # score may be int or float; try int
                try:
                    score = float(row.get("score", ""))
                except Exception:
                    continue
                topic = qid_to_topic.get(qid, "UNKNOWN_TOPIC")
                topic_scores[topic].append(score)

    # Compute statistics
    stats = {}
    for topic, scores in topic_scores.items():
        arr = list(scores)
        cnt = len(arr)
        mean = statistics.mean(arr) if cnt else 0.0
        median = statistics.median(arr) if cnt else 0.0
        var = statistics.pvariance(arr) if cnt else 0.0
        std = statistics.pstdev(arr) if cnt else 0.0
        distribution = {i: 0 for i in range(1, 6)}
        for s in arr:
            try:
                si = int(round(s))
            except Exception:
                continue
            if 1 <= si <= 5:
                distribution[si] += 1

        stats[topic] = {
            "count": cnt,
            "mean": mean,
            "median": median,
            "var": var,
            "std": std,
            "distribution": distribution,
        }

    # Prepare output dir
    os.makedirs(output_dir, exist_ok=True)

    # Write summary CSV
    summary_csv = os.path.join(output_dir, output_file)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "count", "mean", "median", "std", "var", "dist_1", "dist_2", "dist_3", "dist_4", "dist_5"])
        for topic, s in sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True):
            d = s["distribution"]
            writer.writerow([topic, s["count"], f"{s['mean']:.4f}", f"{s['median']:.4f}", f"{s['std']:.4f}", f"{s['var']:.4f}", d[1], d[2], d[3], d[4], d[5]])

    # Plot mean score per topic (bar chart)
    topics_sorted = sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    topics = [t for t, _ in topics_sorted]
    means = [s["mean"] for _, s in topics_sorted]
    counts = [s["count"] for _, s in topics_sorted]

    # Limit label length
    labels = [t if len(t) <= 40 else t[:37] + "..." for t in topics]

    plt.figure(figsize=(12, max(4, len(labels) * 0.2)))
    plt.bar(range(len(means)), means, color="C0")
    plt.xticks(range(len(means)), labels, rotation=90)
    plt.ylabel("Mean score")
    plt.title(f"Mean score per topic: {retrieval_type} + {embedding_model_name}")
    for i, c in enumerate(counts):
        plt.text(i, means[i] + 0.02, str(c), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mean_score_per_topic_{retrieval_type}_{embedding_model_name}.png"), dpi=150)
    plt.close()

    # Boxplot for top-k topics by count
    top_topics = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)[:top_k_topics]
    box_labels = [t for t, _ in top_topics]
    box_scores = [topic_scores[t] for t in box_labels]

    plt.figure(figsize=(12, max(4, len(box_labels) * 0.3)))
    plt.boxplot(box_scores, labels=[(t if len(t) <= 30 else t[:27] + "...") for t in box_labels], showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Score")
    plt.title(f"Score distribution per topic (top {top_k_topics} by count): {retrieval_type} + {embedding_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"score_boxplot_top_topics_{retrieval_type}_{embedding_model_name}.png"), dpi=150)
    plt.close()

    return stats


