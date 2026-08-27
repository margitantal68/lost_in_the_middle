from pathlib import Path

# Use this for LiveRAG dataset
# DATASET = "liverag"
QUESTIONS_PART = 1 # using questions1 (1 context)
# QUESTIONS_PART = 2 # using questions2 (2 contexts)
# BATCH_SIZE = 32


# Use this for SQuAD dataset
DATASET = "squad"
BATCH_SIZE = 64

EXPORT = False
DEFAULT_EXPORT_DIR = Path("embeddings_export")

SQUAD_PATH = "data/squad/dev-v2.0.json"
LIVERAG_PATH = "data/liverag"

# LLM answer generation settings
ANSWER_GENERATION_TEMPERATURE = 0.7

ANSWER_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  #vLLM chat model
ANSWER_MODEL_CODE = "llama"  #vLLM chat model

# ANSWER_MODEL_NAME = "google/gemma-2-9b-it"
# ANSWER_MODEL_CODE = "gemma"  #vLLM chat model

# ANSWER_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# ANSWER_MODEL_CODE = "mistral"  #vLLM chat model

# ANSWER_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# ANSWER_MODEL_CODE = "deepseek"  #vLLM chat model

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"           #vLLM chat model for evaluation

# CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # CrossEncoder model for re-ranking
CROSSENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # CrossEncoder model for re-ranking


# Embedder model settings
model_names = [
    # "PP-MINILM",
    # "BGE-M3",
    # "BGE-SMALL",
    # "BGE-BASE",
    # "BGE-LARGE",
    # "XLMRoBERTaML",
    "E5-BASE"
]

INDEX = 0
EMBEDDER_MODEL_NAME = model_names[INDEX]


# Retrieval settings
# RETRIEVAL_TYPE = "top_k_retrieval"
# RETRIEVAL_TYPE = "hubness_aware_reranking"
RETRIEVAL_TYPE = "crossencoder"

# Context reordering settings
CONTEXT_REORDERED = False
REORDER_POSITION = "reordered_last"  # Options: "reordered_first", "reordered_last", "reordered_middle"

# Answer evaluation settings
START_QUESTION_ID = 0

VERBOSE_ANSWER_GENERATION = False

IS_FOLD = False

fold = "hubness_aware_retrieval_LiveRAG_seed_0_fold_0"
