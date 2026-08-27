# Lost in the Middle (LITM): RAG Evaluation Toolkit

This repository offers a complete toolkit for building, exporting, and evaluating embeddings for the SQuAD 2.0  and LiveRAG datasets. It includes utilities for generating embeddings with Sentence Transformers, exporting them to JSON, and measuring retrieval performance. The repository also provides scripts to generate answers using the llama3.2, gemma2, and Mistral model through vLLM and to evaluate responses using an LLM-as-a-judge approach via vLLM. In addition, it contains all components needed for end-to-end RAG evaluation on SQuAD 2.0 and LiveRAG, including retrievers, rerankers, and evaluation pipelines.


## Features

- Build embeddings for documents and questions using Sentence Transformers.
- Export embeddings in JSON format with metadata.
- Evaluate retrieval performance.
- Generate answers for questions using the llama3.2, gemma2, and Mistral model through vLLM
- Evaluate answers using LLM-as-a-judge method using the qwen2.5 model through vLLM

---

## Installation

This project uses the `uv` package manager for managing dependencies. Follow the steps below to set up the repository:

### Prerequisites

- Python 3.11 or higher
- `uv` package manager ([Install UV](https://uv-pm.netlify.app/))

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/margitantal68/embeddings_hubness.git
   cd embeddings_hubness
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

---
## Project Structure

```
lost_in_the_middle/
├── data/                            # Dataset files
│   ├── answers
│   ├── retriever
│   ├── squad
│   ├── liverag
├── embeddings_export/                # Exported embeddings (ignored in Git)
├── config.py                         # Configuration file for embeddings and evaluation
├── compute_retrieval_metrics.py      # Retrieval metric calculator for CSV outputs
├── eda_liverag.py                    # Exploratory analysis for LiveRAG experiments
├── eda_squad.py                      # Exploratory analysis for SQuAD experiments
├── embeddings_builder.py             # Functions for building and exporting embeddings
├── main_analyze_retrieval_scores.py  # Analysis for retrieval score distributions
├── main_analyze_topic_scores.py      # Topic-level performance analysis
├── main_compare_answer_scores.py     # Compare answer scoring across runs
├── main_prepare_answer_evaluation.py # Prepare data for answer evaluation
├── main_rag_answer_evaluation.py     # RAG answer evaluation pipeline
├── main_rag_answer_generation.py     # RAG answer generation pipeline
├── main_retrieval_evaluation.py     # Embeddings creation and retriever performance evaluation
├── models.py                        # Model wrapper for Sentence Transformers
├── plots.py                         # Plot hubness and precedence distributions
├── retrieval_evaluation.py          # Retrieval evaluation utilities
├── utils.py                         # Helper functions
├── utils_liverag.py                 # LiveRAG-specific utilities
├── pyproject.toml                   # Project configuration
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

---
## Usage

### Retrieval evaluation

   Run the `main_retrieval_evaluation.py` script to evaluate the retrieval for different embeddings.
   ```bash
   python main_retrieval_evaluation.py
   ```
   Embeddings are built and exported if necessary. Retrieval performance (MRR, Recall@k) is evaluated, and hubness metrics are computed.

- Embeddings are exported to the `embeddings_export/` directory if required.
- Context and question embeddings are stored in separate JSON files.
- Retrieved context ids for each question are stored in CSV files under `data/retriever/`.


### Answer generation

* Use the `config.py` file to specify the embeddings and the model for answer generation. 
* Run the `main_answer_generation.py` script to generate answers for different embeddings.
* Context documents are retrieved from the `retriever/` directory and the answers are exported to the `answers/` directory.

### Prepare for RAG (answer) evaluation

   Run the `main_prepare_answer_evaluation.py` script to create the input files for RAG evaluation.
   
### Answer evaluation using LLM-as-a-judge

* Use the `config.py` file to specify the judge model for answer evaluation.
* Run the `main_answer_evaluation.py` script to evaluate the answers.
   ```bash
   python main_answer_evaluation.py
   ```
   Answers are evaluated using the LLM-as-a-judge method.
---
## Notes

- Ensure the required datasets are placed properly in the `data/` directory before running the scripts. 
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)  dataset should be placed in `data/squad/` 
* [LiveRAG](https://huggingface.co/datasets/LiveRAG/Benchmark) dataset should be placed in `data/liverag/`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.