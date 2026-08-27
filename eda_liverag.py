"""
Exploratory Data Analysis (EDA) for the LiveRAG Dataset

This module provides comprehensive analysis of the LiveRAG dataset:
- liverag_contexts.csv: Contains all unique supporting documents
- liverag_questions1.csv: Questions answerable from 1 context
- liverag_questions2.csv: Questions answerable from 2 contexts

The analysis covers:
- Data loading and basic statistics
- Text length and token distributions
- Metadata field analysis (ACS, ACS_Std, IRT-diff, IRT-disc)
- Comparison between single-context and multi-context questions
- Context reusability analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import json
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class LiveRAGEDA:
    """Exploratory Data Analysis for LiveRAG Dataset"""
    
    def __init__(self, data_dir: str = "data/liverag"):
        """
        Initialize the EDA class with dataset paths.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
        self.contexts_file = self.data_dir / "liverag_contexts.csv"
        self.questions1_file = self.data_dir / "liverag_questions1.csv"
        self.questions2_file = self.data_dir / "liverag_questions2.csv"
        
        # DataFrames will be loaded lazily
        self.contexts = None
        self.questions1 = None
        self.questions2 = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading LiveRAG dataset...")
        self.contexts = pd.read_csv(self.contexts_file)
        self.questions1 = pd.read_csv(self.questions1_file)
        self.questions2 = pd.read_csv(self.questions2_file)
        print(f"✓ Loaded {len(self.contexts)} contexts")
        print(f"✓ Loaded {len(self.questions1)} single-context questions")
        print(f"✓ Loaded {len(self.questions2)} multi-context questions")
        
    def print_basic_info(self):
        """Print basic information about the datasets"""
        print("\n" + "="*80)
        print("BASIC DATASET INFORMATION")
        print("="*80)
        
        print(f"\nTotal samples: {len(self.questions1) + len(self.questions2)}")
        print(f"  - Single-context questions (Q1): {len(self.questions1)}")
        print(f"  - Multi-context questions (Q2): {len(self.questions2)}")
        print(f"  - Total unique contexts: {len(self.contexts)}")
        
        print("\n" + "-"*80)
        print("CONTEXTS")
        print("-"*80)
        print(self.contexts.head())
        
        print("\n" + "-"*80)
        print("QUESTIONS (1 context)")
        print("-"*80)
        print(self.questions1.head())
        
        print("\n" + "-"*80)
        print("QUESTIONS (2 contexts)")
        print("-"*80)
        print(self.questions2.head())
        
    def analyze_text_lengths(self):
        """Analyze text length statistics"""
        print("\n" + "="*80)
        print("TEXT LENGTH ANALYSIS")
        print("="*80)
        
        # Context lengths
        self.contexts['text_length'] = self.contexts['context_text'].str.len()
        self.contexts['word_count'] = self.contexts['context_text'].str.split().str.len()
        
        print("\n[CONTEXTS]")
        print(f"  Total characters: {self.contexts['text_length'].sum():,.0f}")
        print(f"  Mean length: {self.contexts['text_length'].mean():.1f} chars")
        print(f"  Median length: {self.contexts['text_length'].median():.1f} chars")
        print(f"  Min/Max: {self.contexts['text_length'].min()}/{self.contexts['text_length'].max()}")
        print(f"  Std dev: {self.contexts['text_length'].std():.1f}")
        print(f"  Mean word count: {self.contexts['word_count'].mean():.1f} words")
        
        # Question lengths (Q1)
        self.questions1['question_length'] = self.questions1['question_text'].str.len()
        self.questions1['answer_length'] = self.questions1['answer_text'].str.len()
        self.questions1['question_words'] = self.questions1['question_text'].str.split().str.len()
        self.questions1['answer_words'] = self.questions1['answer_text'].str.split().str.len()
        
        print("\n[QUESTIONS (1 context)]")
        print(f"  Mean question length: {self.questions1['question_length'].mean():.1f} chars")
        print(f"  Mean answer length: {self.questions1['answer_length'].mean():.1f} chars")
        print(f"  Mean question words: {self.questions1['question_words'].mean():.1f} words")
        print(f"  Mean answer words: {self.questions1['answer_words'].mean():.1f} words")
        
        # Question lengths (Q2)
        self.questions2['question_length'] = self.questions2['question_text'].str.len()
        self.questions2['answer_length'] = self.questions2['answer_text'].str.len()
        self.questions2['question_words'] = self.questions2['question_text'].str.split().str.len()
        self.questions2['answer_words'] = self.questions2['answer_text'].str.split().str.len()
        
        print("\n[QUESTIONS (2 contexts)]")
        print(f"  Mean question length: {self.questions2['question_length'].mean():.1f} chars")
        print(f"  Mean answer length: {self.questions2['answer_length'].mean():.1f} chars")
        print(f"  Mean question words: {self.questions2['question_words'].mean():.1f} words")
        print(f"  Mean answer words: {self.questions2['answer_words'].mean():.1f} words")
        
        return {
            'contexts': self.contexts[['context_id', 'text_length', 'word_count']],
            'questions1': self.questions1[['question_id', 'question_length', 'answer_length', 
                                          'question_words', 'answer_words']],
            'questions2': self.questions2[['question_id', 'question_length', 'answer_length',
                                          'question_words', 'answer_words']],
        }
    
    def analyze_metadata_fields(self):
        """Analyze metadata fields: ACS, ACS_Std, IRT-diff, IRT-disc"""
        print("\n" + "="*80)
        print("METADATA FIELD ANALYSIS")
        print("="*80)
        
        metadata_fields = ['ACS', 'ACS_Std', 'IRT-diff', 'IRT-disc']
        
        print("\n[QUESTIONS (1 context)]")
        for field in metadata_fields:
            if field in self.questions1.columns:
                col_data = self.questions1[field]
                print(f"\n  {field}:")
                print(f"    Mean: {col_data.mean():.4f}")
                print(f"    Median: {col_data.median():.4f}")
                print(f"    Std: {col_data.std():.4f}")
                print(f"    Min/Max: {col_data.min():.4f} / {col_data.max():.4f}")
                print(f"    Missing: {col_data.isna().sum()}")
        
        print("\n[QUESTIONS (2 contexts)]")
        for field in metadata_fields:
            if field in self.questions2.columns:
                col_data = self.questions2[field]
                print(f"\n  {field}:")
                print(f"    Mean: {col_data.mean():.4f}")
                print(f"    Median: {col_data.median():.4f}")
                print(f"    Std: {col_data.std():.4f}")
                print(f"    Min/Max: {col_data.min():.4f} / {col_data.max():.4f}")
                print(f"    Missing: {col_data.isna().sum()}")
        
        return {
            'questions1': self.questions1[['question_id'] + metadata_fields],
            'questions2': self.questions2[['question_id'] + metadata_fields],
        }
    
    def analyze_context_reusability(self):
        """Analyze how many times each context is used across questions"""
        print("\n" + "="*80)
        print("CONTEXT REUSABILITY ANALYSIS")
        print("="*80)
        
        # Count context usage in Q1
        context_usage_q1 = self.questions1['context_id'].value_counts().to_dict()
        
        # Count context usage in Q2
        # Note: context_ids in Q2 is a string representation like "[1, 2]"
        context_usage_q2 = Counter()
        for ctx_ids_str in self.questions2['context_ids']:
            # Parse the string representation of list
            try:
                ctx_ids = eval(ctx_ids_str) if isinstance(ctx_ids_str, str) else ctx_ids_str
                for ctx_id in ctx_ids:
                    context_usage_q2[ctx_id] += 1
            except:
                pass
        
        # Combine usage counts
        total_usage = Counter(context_usage_q1)
        for ctx_id, count in context_usage_q2.items():
            total_usage[ctx_id] += count
        
        print(f"\n  Contexts used in Q1: {len(context_usage_q1)}")
        print(f"  Contexts used in Q2: {len(set(context_usage_q2.keys()))}")
        print(f"  Total unique contexts used: {len(total_usage)}")
        print(f"  Contexts never used: {len(self.contexts) - len(total_usage)}")
        
        print("\n  Usage distribution:")
        print(f"    Mean usage per context: {np.mean(list(total_usage.values())):.2f}")
        print(f"    Median usage: {np.median(list(total_usage.values())):.1f}")
        print(f"    Max usage: {max(total_usage.values())}")
        print(f"    Min usage: {min(total_usage.values())}")
        
        # Most used contexts
        print("\n  Top 10 most reused contexts:")
        for ctx_id, count in total_usage.most_common(10):
            ctx_text = self.contexts[self.contexts['context_id'] == ctx_id]['context_text'].values
            if len(ctx_text) > 0:
                preview = ctx_text[0][:80].replace('\n', ' ')
                print(f"    Context {ctx_id}: {count} questions - '{preview}...'")
        
        return {
            'context_usage_q1': context_usage_q1,
            'context_usage_q2': context_usage_q2,
            'total_usage': total_usage,
        }
    
    def compare_question_types(self):
        """Compare characteristics between Q1 and Q2"""
        print("\n" + "="*80)
        print("COMPARISON: SINGLE-CONTEXT vs MULTI-CONTEXT QUESTIONS")
        print("="*80)
        
        comparison = pd.DataFrame({
            'Metric': [
                'Sample Count',
                'Avg Question Length (chars)',
                'Avg Answer Length (chars)',
                'Avg Question Words',
                'Avg Answer Words',
                'Mean ACS',
                'Mean ACS_Std',
                'Mean IRT-diff',
                'Mean IRT-disc',
            ],
            'Q1 (1 context)': [
                len(self.questions1),
                self.questions1['question_length'].mean(),
                self.questions1['answer_length'].mean(),
                self.questions1['question_words'].mean(),
                self.questions1['answer_words'].mean(),
                self.questions1['ACS'].mean(),
                self.questions1['ACS_Std'].mean(),
                self.questions1['IRT-diff'].mean(),
                self.questions1['IRT-disc'].mean(),
            ],
            'Q2 (2 contexts)': [
                len(self.questions2),
                self.questions2['question_length'].mean(),
                self.questions2['answer_length'].mean(),
                self.questions2['question_words'].mean(),
                self.questions2['answer_words'].mean(),
                self.questions2['ACS'].mean(),
                self.questions2['ACS_Std'].mean(),
                self.questions2['IRT-diff'].mean(),
                self.questions2['IRT-disc'].mean(),
            ]
        })
        
        print()
        print(comparison.to_string(index=False))
        
        return comparison
    
    def plot_distributions(self, save_dir: str = "plots"):
        """Generate distribution plots"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Context length distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(self.contexts['text_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Context Length (characters)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Context Lengths')
        axes[0].axvline(self.contexts['text_length'].mean(), color='r', 
                       linestyle='--', label=f"Mean: {self.contexts['text_length'].mean():.0f}")
        axes[0].legend()
        
        axes[1].hist(self.contexts['word_count'], bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Context Length (words)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Context Word Counts')
        axes[1].axvline(self.contexts['word_count'].mean(), color='r', 
                       linestyle='--', label=f"Mean: {self.contexts['word_count'].mean():.0f}")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "01_context_lengths.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 01_context_lengths.png")
        plt.close()
        
        # Plot 2: Question characteristics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Question lengths
        axes[0, 0].hist([self.questions1['question_length'], self.questions2['question_length']], 
                       label=['Q1', 'Q2'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Question Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Question Length Distribution')
        axes[0, 0].legend()
        
        # Answer lengths
        axes[0, 1].hist([self.questions1['answer_length'], self.questions2['answer_length']], 
                       label=['Q1', 'Q2'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Answer Length (characters)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Answer Length Distribution')
        axes[0, 1].legend()
        
        # Question word counts
        axes[1, 0].hist([self.questions1['question_words'], self.questions2['question_words']], 
                       label=['Q1', 'Q2'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Question Word Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Question Word Count Distribution')
        axes[1, 0].legend()
        
        # Answer word counts
        axes[1, 1].hist([self.questions1['answer_words'], self.questions2['answer_words']], 
                       label=['Q1', 'Q2'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Answer Word Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Answer Word Count Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "02_question_characteristics.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 02_question_characteristics.png")
        plt.close()
        
        # Plot 3: Metadata fields
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metadata_fields = ['ACS', 'ACS_Std', 'IRT-diff', 'IRT-disc']
        for idx, field in enumerate(metadata_fields):
            ax = axes[idx // 2, idx % 2]
            
            data_q1 = self.questions1[field].dropna()
            data_q2 = self.questions2[field].dropna()
            
            ax.hist([data_q1, data_q2], label=['Q1', 'Q2'], bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(field)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {field}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "03_metadata_distributions.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 03_metadata_distributions.png")
        plt.close()
        
        # Plot 4: Box plots for metadata comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, field in enumerate(metadata_fields):
            ax = axes[idx // 2, idx % 2]
            
            data_to_plot = [
                self.questions1[field].dropna(),
                self.questions2[field].dropna()
            ]
            
            bp = ax.boxplot(data_to_plot, labels=['Q1', 'Q2'], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_ylabel(field)
            ax.set_title(f'Box Plot: {field}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "04_metadata_boxplots.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 04_metadata_boxplots.png")
        plt.close()
        
        # Plot 5: Context reusability
        context_usage_q1_dict = self.questions1['context_id'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(context_usage_q1_dict.values, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Number of Questions per Context')
        axes[0].set_ylabel('Frequency (Q1)')
        axes[0].set_title('Context Reusability in Q1')
        axes[0].set_yscale('log')
        
        # Context usage distribution (top used)
        top_contexts = context_usage_q1_dict.head(20)
        axes[1].barh(range(len(top_contexts)), top_contexts.values)
        axes[1].set_yticks(range(len(top_contexts)))
        axes[1].set_yticklabels([f'Context {cid}' for cid in top_contexts.index])
        axes[1].set_xlabel('Number of Questions')
        axes[1].set_title('Top 20 Most Reused Contexts (Q1)')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path / "05_context_reusability.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: 05_context_reusability.png")
        plt.close()
        
        print(f"\n  All visualizations saved to: {save_path.absolute()}")
    
    def generate_summary_stats(self) -> dict:
        """Generate comprehensive summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        summary = {
            'dataset_overview': {
                'total_questions': len(self.questions1) + len(self.questions2),
                'single_context_questions': len(self.questions1),
                'multi_context_questions': len(self.questions2),
                'total_unique_contexts': len(self.contexts),
            },
            'contexts': {
                'mean_length_chars': self.contexts['text_length'].mean(),
                'median_length_chars': self.contexts['text_length'].median(),
                'mean_word_count': self.contexts['word_count'].mean(),
                'total_characters': self.contexts['text_length'].sum(),
            },
            'questions1': {
                'question_length_mean': self.questions1['question_length'].mean(),
                'answer_length_mean': self.questions1['answer_length'].mean(),
                'question_words_mean': self.questions1['question_words'].mean(),
                'answer_words_mean': self.questions1['answer_words'].mean(),
            },
            'questions2': {
                'question_length_mean': self.questions2['question_length'].mean(),
                'answer_length_mean': self.questions2['answer_length'].mean(),
                'question_words_mean': self.questions2['question_words'].mean(),
                'answer_words_mean': self.questions2['answer_words'].mean(),
            },
        }
        
        print("\nDataset Overview:")
        for key, value in summary['dataset_overview'].items():
            print(f"  {key}: {value}")
        
        return summary


def main():
    """Run complete EDA pipeline"""
    print("\n" + "="*80)
    print("LIVERAG EXPLORATORY DATA ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize and load data
    eda = LiveRAGEDA(data_dir="data/liverag")
    eda.load_data()
    
    # Basic information
    eda.print_basic_info()
    
    # Detailed analyses
    eda.analyze_text_lengths()
    eda.analyze_metadata_fields()
    eda.analyze_context_reusability()
    eda.compare_question_types()
    
    # Generate visualizations
    eda.plot_distributions(save_dir="plots/liverag_eda")
    
    # Summary statistics
    summary = eda.generate_summary_stats()
    
    # Save summary to JSON
    with open("data/liverag/summary_statistics.json", "w") as f:
        # Convert numpy types to Python types for JSON serialization
        summary_serializable = {}
        for key, val in summary.items():
            if isinstance(val, dict):
                summary_serializable[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                    for k, v in val.items()
                }
            else:
                summary_serializable[key] = val
        
        json.dump(summary_serializable, f, indent=2)
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    print("\nGenerated outputs:")
    print("  ✓ Console summary")
    print("  ✓ Visualizations in plots/liverag_eda/")
    print("  ✓ Summary statistics saved to data/liverag/summary_statistics.json")


if __name__ == "__main__":
    main()
