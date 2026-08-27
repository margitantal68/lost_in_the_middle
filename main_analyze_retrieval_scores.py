import pandas as pd
import numpy as np

from config import DATASET, EMBEDDER_MODEL_NAME, RETRIEVAL_TYPE, ANSWER_MODEL_CODE

def analyze_retrieval_scores(retrieval_file, evaluation_file):
    """
    Analyze score statistics based on ground truth context position in retrieval results.
    
    Args:
        retrieval_file (str): Path to CSV file with retrieval results (no header).
                             Columns: question_id, groundtruth_ctx_id, ctx_id_1, ctx_id_2, 
                                     ctx_id_3, ctx_id_4, ctx_id_5
        evaluation_file (str): Path to CSV file with evaluation results (with header).
                              Columns: question_id, question, answer, llm_answer, score, explanation
    
    Returns:
        dict: Dictionary containing statistics for three categories
    """
    
    # Read the CSV files
    retrieval_df = pd.read_csv(retrieval_file, header=None)
    evaluation_df = pd.read_csv(evaluation_file)
    
    # Name the columns for retrieval data
    retrieval_df.columns = ['question_id', 'groundtruth_ctx_id', 'ctx_id_1', 'ctx_id_2', 
                            'ctx_id_3', 'ctx_id_4', 'ctx_id_5']
    
    # Merge the two dataframes on question_id
    merged_df = pd.merge(retrieval_df, evaluation_df, on='question_id')
    
    # Categorize questions based on ground truth position
    def categorize_position(row):
        groundtruth_id = row['groundtruth_ctx_id']
        
        # Check if groundtruth is in top 3 positions
        if groundtruth_id in [row['ctx_id_1']]:
            return 'first'
        elif groundtruth_id in [row['ctx_id_2']]:
            return 'second'
        elif groundtruth_id in [row['ctx_id_3']]:
            return 'third'
        elif groundtruth_id in [row['ctx_id_4']]:
            return 'fourth'
        elif groundtruth_id in [row['ctx_id_5']]:
            return 'fifth'
         # If not found in any retrieved context
        else:
            return 'not_retrieved'
        
            # if groundtruth_id in [row['ctx_id_1'], row['ctx_id_2'], row['ctx_id_3']]:
            #     return 'top_3'
            # # Check if groundtruth is in last 2 positions
            # elif groundtruth_id in [row['ctx_id_4'], row['ctx_id_5']]:
            #     return 'last_2'
       
    
    merged_df['position_category'] = merged_df.apply(categorize_position, axis=1)
    
    # Compute statistics for each category
    categories = {
        'first' : 'Ground truth in first position (ctx_id_1)',
        'second': 'Ground truth in second position (ctx_id_2)',
        'third' : 'Ground truth in third position (ctx_id_3)',
        'fourth': 'Ground truth in fourth position (ctx_id_4)',
        'fifth' : 'Ground truth in fifth position (ctx_id_5)',
        'not_retrieved': 'Ground truth not retrieved'
    }
    
    results = {}
    
    for category_key, category_desc in categories.items():
        category_scores = merged_df[merged_df['position_category'] == category_key]['score']
        
        if len(category_scores) > 0:
            stats = {
                'category': category_desc,
                'count': len(category_scores),
                'mean': category_scores.mean(),
                'std': category_scores.std(),
                'min': category_scores.min(),
                'max': category_scores.max(),
                'median': category_scores.median(),
                'q25': category_scores.quantile(0.25),
                'q75': category_scores.quantile(0.75),
            }
        else:
            stats = {
                'category': category_desc,
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'q25': None,
                'q75': None,
            }
        
        results[category_key] = stats
    
    # Print statistics
    print("=" * 80)
    print("RETRIEVAL SCORE ANALYSIS")
    print("=" * 80)
    
    for category_key in ['first', 'second', 'third', 'fourth', 'fifth', 'not_retrieved']:
        stats = results[category_key]
        print(f"\n{stats['category']}")
        print("-" * 80)
        print(f"  Count:       {stats['count']}")
        
        if stats['count'] > 0:
            print(f"  Mean:        {stats['mean']:.2f}")
            print(f"  Std Dev:     {stats['std']:.2f}")
            print(f"  Min:         {stats['min']:.0f}")
            print(f"  Max:         {stats['max']:.0f}")
            print(f"  Median:      {stats['median']:.2f}")
            print(f"  Q1 (25%):    {stats['q25']:.2f}")
            print(f"  Q3 (75%):    {stats['q75']:.2f}")
        else:
            print(f"  No data available")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    print(f"Analyzing retrieval scores for {RETRIEVAL_TYPE} with {EMBEDDER_MODEL_NAME} on {DATASET} dataset...")
    # retrieval_file = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv"
    # evaluation_file = f"data/answers/_evaluated_output_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{ANSWER_MODEL_CODE}.csv"
    
    retrieval_file = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_1.csv"
    
    evaluation_file = "data/answers/_evaluated_output_answers_liverag_crossencoder_BGE-M3_mistral.csv"
    # evaluation_file = f"data/answers/evaluated_output_answers_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{ANSWER_MODEL_CODE}.csv"
    # evaluation_file = f"data/answers/_evaluated_output_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{ANSWER_MODEL_CODE}.csv"
    print(f"Analyzing retrieval scores for {RETRIEVAL_TYPE} with {EMBEDDER_MODEL_NAME}...")
    results = analyze_retrieval_scores(retrieval_file, evaluation_file)
