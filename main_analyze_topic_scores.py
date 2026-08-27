from utils import squad_statistics, compute_score_distribution
from utils import export_squad_topics_csv, compute_topic_score_statistics

if __name__ == "__main__":
    # squad_statistics()
    # FILE = "data/answers/evaluated_output_answers_top_k_retrieval_BGE-M3.csv"
    # FILE = "data/answers/evaluated_output_answers_hubness_aware_reranking_BGE-M3.csv"
    # FILE = "data/answers/evaluated_output_answers_crossencoder_BGE-M3.csv"

    # FILE = "data/answers/evaluated_output_answers_top_k_retrieval_E5-BASE.csv"
    # FILE = "data/answers/evaluated_output_answers_hubness_aware_reranking_E5-BASE.csv"
    # FILE = "data/answers/evaluated_output_answers_crossencoder_E5-BASE.csv"
    
    # compute_score_distribution(csv_file_path=FILE, score_column="score")

    # export_squad_topics_csv()

    # EMBEDDING_MODEL_NAME = "E5-BASE"
    EMBEDDING_MODEL_NAME = "BGE-M3"

    
    # RETRIEVAL_TYPE = "top_k_retrieval"
    # RETRIEVAL_TYPE = "hubness_aware_reranking"
    RETRIEVAL_TYPE = "crossencoder"

    INPUT_FILE = f"data/answers/evaluated_output_answers_{RETRIEVAL_TYPE}_{EMBEDDING_MODEL_NAME}.csv"
    OUTPUT_DIR = "data/answers/topics/"
    OUTPUT_FILE = f"topic_stats_{RETRIEVAL_TYPE}_{EMBEDDING_MODEL_NAME}.csv"
    compute_topic_score_statistics(RETRIEVAL_TYPE, EMBEDDING_MODEL_NAME, top_k_topics=30)