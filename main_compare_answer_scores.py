import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

seeds = [0, 1, 42, 271, 314]
FILE3 = "data/answers/evaluated_output_answers_squad_top_k_retrieval_BGE-M3_newprompt.csv"
SUMMARY_PATH = Path("data/answers/seed_score_change_summary.csv")
PLOT_PATH_STACK = Path("plots/score_change_stack_by_seed.png")
PLOT_PATH_DELTA = Path("plots/score_delta_and_improvement_by_seed.png")


def load_answer_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'question_Id' in df.columns and 'question_id' not in df.columns:
        df = df.rename(columns={'question_Id': 'question_id'})
    if 'score' not in df.columns:
        raise ValueError(f"Missing required 'score' column in {path}")
    return df


def normalize_question_id(df: pd.DataFrame) -> pd.DataFrame:
    if df['question_id'].dtype == object:
        if df['question_id'].str.isnumeric().all():
            df['question_id'] = df['question_id'].astype(int)
        else:
            df['question_id'] = df['question_id'].astype(str)
    return df


def describe_scores(scores: pd.Series) -> dict:
    scores = pd.to_numeric(scores, errors='coerce')
    return {
        'count': int(scores.count()),
        'mean': float(scores.mean()),
        'median': float(scores.median()),
        'std': float(scores.std()),
        'min': float(scores.min()),
        'max': float(scores.max()),
    }


def print_summary(label: str, stats: dict) -> None:
    print(f"\n{label} score statistics:")
    for name, value in stats.items():
        print(f"  {name:>6}: {value:.4f}" if isinstance(value, float) else f"  {name:>6}: {value}")


def compute_seed_results(seed: int, original_df: pd.DataFrame) -> dict:
    file1 = f"data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_{seed}_fold_0.csv"
    file2 = f"data/answers/evaluated_output_answers_hubness_aware_retrieval_SQuAD_seed_{seed}_fold_1.csv"
    file1_df = normalize_question_id(load_answer_file(file1))
    file2_df = normalize_question_id(load_answer_file(file2))
    modified_df = pd.concat([file1_df, file2_df], ignore_index=True)
    modified_df = modified_df.sort_values('question_id', kind='mergesort').reset_index(drop=True)

    original_scores = original_df.set_index('question_id')['score'].rename('original_score')
    merged_df = modified_df.set_index('question_id').join(original_scores, how='left', validate='one_to_one').reset_index()

    if merged_df['original_score'].isna().any():
        missing = merged_df.loc[merged_df['original_score'].isna(), 'question_id'].unique()
        raise ValueError(f"Missing original scores for question_ids: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    merged_df['modified_score'] = pd.to_numeric(merged_df['score'], errors='coerce')
    merged_df['original_score'] = pd.to_numeric(merged_df['original_score'], errors='coerce')
    merged_df['score_delta'] = merged_df['modified_score'] - merged_df['original_score']

    improved = int((merged_df['score_delta'] > 0).sum())
    unchanged = int((merged_df['score_delta'] == 0).sum())
    declined = int((merged_df['score_delta'] < 0).sum())
    total = len(merged_df)

    return {
        'seed': seed,
        'seed_label': f"seed {seed}",
        'total_questions': total,
        'improved': improved,
        'unchanged': unchanged,
        'declined': declined,
        'improved_pct': improved / total if total else 0.0,
        'unchanged_pct': unchanged / total if total else 0.0,
        'declined_pct': declined / total if total else 0.0,
        'avg_delta': float(merged_df['score_delta'].mean()),
        'median_delta': float(merged_df['score_delta'].median()),
        'modified_mean': float(merged_df['modified_score'].mean()),
        'original_mean': float(merged_df['original_score'].mean()),
        'delta_stats': describe_scores(merged_df['score_delta']),
        'modified_stats': describe_scores(merged_df['modified_score']),
        'original_stats': describe_scores(merged_df['original_score']),
        'merged_df': merged_df,
    }


def plot_seed_count_stack(stats_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = stats_df['seed_label'].tolist()
    improved = stats_df['improved'].tolist()
    unchanged = stats_df['unchanged'].tolist()
    declined = stats_df['declined'].tolist()

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.bar(labels, improved, color='green', label='Improved')
    ax.bar(labels, unchanged, bottom=improved, color='gray', label='Unchanged')
    bottom = [i + u for i, u in zip(improved, unchanged)]
    ax.bar(labels, declined, bottom=bottom, color='red', label='Declined')
    ax.set_ylabel('# questions')
    ax.set_title('Score change counts by seed relative to original top-k method')
    ax.legend()

    for i, (imp, unch, dec) in enumerate(zip(improved, unchanged, declined)):
        if imp > 0:
            ax.text(i, imp / 2, f"{imp}\n{stats_df.loc[i, 'improved_pct']:.1%}", ha='center', va='center', color='white', fontsize=9)
        if unch > 0:
            ax.text(i, imp + unch / 2, f"{unch}\n{stats_df.loc[i, 'unchanged_pct']:.1%}", ha='center', va='center', color='white', fontsize=9)
        if dec > 0:
            ax.text(i, imp + unch + dec / 2, f"{dec}\n{stats_df.loc[i, 'declined_pct']:.1%}", ha='center', va='center', color='white', fontsize=9)

    fig.savefig(out_path)
    print(f"Plot saved to: {out_path}")


def plot_method_score_comparison(stats_df: pd.DataFrame, original_avg: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["top-k"] + stats_df['seed_label'].tolist()
    scores = [original_avg] + stats_df['modified_mean'].tolist()
    deltas = [0.0] + (stats_df['modified_mean'] - original_avg).tolist()
    colors = ["gray"] + ["green"] * len(stats_df)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = range(len(labels))
    ax.bar(x, scores, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Average score')
    ax.set_title('Average score by method: top-k vs hubness-aware seeds')

    for i, (score, delta) in enumerate(zip(scores, deltas)):
        label = f"{score:.3f}"
        if i > 0:
            label += f"\n(+{delta:.3f})" if delta >= 0 else f"\n({delta:.3f})"
        ax.text(i, score + 0.03, label, ha='center', va='bottom', fontsize=9, color='black')

    fig.savefig(out_path)
    print(f"Plot saved to: {out_path}")


def compute_deltas_by_seed(results: list[dict]) -> pd.DataFrame:
    """
    Extract delta information for each seed: question_id, original, modified, delta, seed
    
    Args:
        results: List of result dictionaries from compute_seed_results
        
    Returns:
        DataFrame with columns: question_id, original_score, modified_score, score_delta, seed
    """
    deltas_list = []
    for result in results:
        df = result['merged_df'][['question_id', 'original_score', 'modified_score', 'score_delta']].copy()
        df['seed'] = result['seed']
        deltas_list.append(df)
    
    return pd.concat(deltas_list, ignore_index=True)


def plot_delta_distribution(results: list[dict], out_path: Path = None) -> pd.DataFrame:
    """
    Plot the distribution of score deltas (changes between original and modified scores).
    
    Args:
        results: List of result dictionaries from compute_seed_results
        out_path: Optional path to save the plot. If None, uses default path.
        
    Returns:
        DataFrame with delta information
    """
    if out_path is None:
        out_path = Path("plots/delta_distribution.png")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute deltas by seed
    deltas_df = compute_deltas_by_seed(results)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # 1. Histogram of all deltas across all seeds
    ax = axes[0, 0]
    all_deltas = deltas_df['score_delta'].values
    ax.hist(all_deltas, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=all_deltas.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {all_deltas.mean():.3f}')
    ax.set_xlabel('Score Delta (Modified - Original)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Score Deltas (All Seeds Combined)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Histogram of deltas by seed (overlapped)
    ax = axes[0, 1]
    for seed in sorted(deltas_df['seed'].unique()):
        seed_deltas = deltas_df[deltas_df['seed'] == seed]['score_delta'].values
        ax.hist(seed_deltas, bins=20, alpha=0.5, label=f'Seed {seed}')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Score Delta (Modified - Original)')
    ax.set_ylabel('Frequency')
    ax.set_title('Delta Distribution by Seed (Overlapped)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Box plot of deltas by seed
    ax = axes[1, 0]
    seed_data = [deltas_df[deltas_df['seed'] == seed]['score_delta'].values for seed in sorted(deltas_df['seed'].unique())]
    bp = ax.boxplot(seed_data, labels=[f"Seed {seed}" for seed in sorted(deltas_df['seed'].unique())], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Score Delta')
    ax.set_title('Delta Distribution by Seed (Box Plot)')
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    summary_stats = []
    for seed in sorted(deltas_df['seed'].unique()):
        seed_deltas = deltas_df[deltas_df['seed'] == seed]['score_delta']
        improved = int((seed_deltas > 0).sum())
        unchanged = int((seed_deltas == 0).sum())
        declined = int((seed_deltas < 0).sum())
        summary_stats.append([
            f"Seed {seed}",
            f"{improved}",
            f"{unchanged}",
            f"{declined}",
            f"{seed_deltas.mean():.3f}",
            f"{seed_deltas.median():.3f}",
        ])
    
    table = ax.table(
        cellText=summary_stats,
        colLabels=['Seed', 'Improved', 'Unchanged', 'Declined', 'Mean Δ', 'Median Δ'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_stats) + 1):
        for j in range(6):
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
    
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"Delta distribution plot saved to: {out_path}")
    
    return deltas_df


def main() -> None:
    original_df = normalize_question_id(load_answer_file(FILE3))
    results = []
    merged_results = []

    for seed in seeds:
        stats = compute_seed_results(seed, original_df)
        results.append(stats)
        merged_results.append(stats['merged_df'].assign(seed=seed))

        print(f"\n===== Seed {seed} =====")
        print_summary('Modified (FILE1 + FILE2)', stats['modified_stats'])
        print_summary('Original (FILE3)', stats['original_stats'])
        print_summary('Delta (Modified - Original)', stats['delta_stats'])
        print(f"Score change counts: improved={stats['improved']}, unchanged={stats['unchanged']}, declined={stats['declined']}")
        print(f"Improvement rate: {stats['improved_pct']:.2%}")
        print(f"Average delta: {stats['avg_delta']:.4f}")

    summary_df = pd.DataFrame([
        {
            'seed': stat['seed'],
            'seed_label': stat['seed_label'],
            'total_questions': stat['total_questions'],
            'improved': stat['improved'],
            'unchanged': stat['unchanged'],
            'declined': stat['declined'],
            'improved_pct': stat['improved_pct'],
            'unchanged_pct': stat['unchanged_pct'],
            'declined_pct': stat['declined_pct'],
            'avg_delta': stat['avg_delta'],
            'median_delta': stat['median_delta'],
            'modified_mean': stat['modified_mean'],
            'original_mean': stat['original_mean'],
        }
        for stat in results
    ])

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSummary saved to: {SUMMARY_PATH}")

    plot_seed_count_stack(summary_df, PLOT_PATH_STACK)
    plot_method_score_comparison(summary_df, original_df['score'].mean(), PLOT_PATH_DELTA)
    
    # Plot delta distribution
    deltas_df = plot_delta_distribution(results, Path("plots/delta_distribution.png"))
    deltas_out_path = Path('data/answers/deltas_by_seed_all.csv')
    deltas_df.to_csv(deltas_out_path, index=False)
    print(f"Delta information saved to: {deltas_out_path}")

    merged_all = pd.concat(merged_results, ignore_index=True)
    merged_out_path = Path('data/answers/compared_scores_FILE3_vs_baseline_all_seeds.csv')
    merged_all.to_csv(merged_out_path, index=False)
    print(f"All seed merged comparison saved to: {merged_out_path}")


if __name__ == '__main__':
    main()
