import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Any
from pathlib import Path
from sentence_transformers import CrossEncoder

from models import SentenceTransformerEmbedder
from retrieval_evaluation import evaluate_retrieval, evaluate_retrieval_2contexts, evaluate_retrieval_with_crossencoder
from embeddings_builder import build_and_export

from utils import (
    extract_squad_data,
    get_or_create_context_collection,
)

from config import EXPORT, QUESTIONS_PART, SQUAD_PATH, LIVERAG_PATH, DATASET
from config import EMBEDDER_MODEL_NAME, INDEX, DEFAULT_EXPORT_DIR, RETRIEVAL_TYPE
from config import CROSSENCODER_MODEL_NAME, ANSWER_MODEL_CODE


models = [
    # SentenceTransformerEmbedder('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    # SentenceTransformerEmbedder("BAAI/bge-m3"),
    # SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
    # SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5"),
    # SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5"),
    # SentenceTransformerEmbedder("sentence-transformers/paraphrase-xlm-r-multilingual-v1"),
    SentenceTransformerEmbedder("intfloat/multilingual-e5-base"),
]


def evaluate_squad():
    CONTEXT_EXPORT_NAME  = f"squad_contexts_{EMBEDDER_MODEL_NAME}"
    QUESTION_EXPORT_NAME = f"squad_questions_{EMBEDDER_MODEL_NAME}"
   
    model = models[INDEX]
    print(f"Loading model: {EMBEDDER_MODEL_NAME}")

    print(f"\nParsing SQuAD at: {SQUAD_PATH}")
    contexts, questions = extract_squad_data(SQUAD_PATH)

    print(f"Extracted {len(contexts)} unique contexts and {len(questions)} questions.")

    # questions_dict: {question_id: (question_text, context_id)}
    question_ids = list(questions.keys())
    question_context_ids = [str(questions[qid][1]) for qid in question_ids]
    print("Length of question_context_ids:", len(question_context_ids))


    # --------------------------------------------------------
    # Build + export context embeddings
    # --------------------------------------------------------
    print("\nBuilding context embeddings...")


    EXPORT = False
    print(f"EXPORT set to: {EXPORT}")

    context_ids = list(contexts.keys())
    context_texts = list(contexts.values())
    
    # Build + export
    _, context_embeddings = build_and_export(
        context_ids,
        context_texts,
        model,
        name=CONTEXT_EXPORT_NAME,
        export=EXPORT,
        contextids=None
    )

    # --------------------------------------------------------
    # Build + export question embeddings
    # --------------------------------------------------------
    print("\nBuilding question embeddings...")
    question_ids = list(questions.keys())
    question_texts = [q[0] for q in questions.values()]


    # Build + export
    _, question_embeddings = build_and_export(
        question_ids,
        question_texts,
        model,
        name=QUESTION_EXPORT_NAME,
        export=EXPORT,
        contextids=question_context_ids
    )

    if EXPORT:
        print("\nAll work complete. Embeddings exported to:")
        print(f"  {DEFAULT_EXPORT_DIR}/{CONTEXT_EXPORT_NAME}.json")
        print(f"  {DEFAULT_EXPORT_DIR}/{QUESTION_EXPORT_NAME}.json")

    persist_dir = f"chroma_db_{DATASET}_{EMBEDDER_MODEL_NAME.lower()}"

    context_collection = get_or_create_context_collection(
        collection_name=CONTEXT_EXPORT_NAME,
        ids=context_ids,
        texts=context_texts,
        embeddings=context_embeddings,
        persist_dir=persist_dir
    )
    
    # --------------------------------------------------------
    # Evaluate retrieval
    # --------------------------------------------------------

    evaluate_retrieval(
        question_embeddings=question_embeddings,
        question_context_ids=question_context_ids,
        context_collection=context_collection,  
        question_ids=question_ids,
        top_k=5,
        output_csv=f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv"
    )


def evaluate_squad_crossencoder():
    CONTEXT_EXPORT_NAME = f"squad_contexts_{EMBEDDER_MODEL_NAME}"
    QUESTION_EXPORT_NAME = f"squad_questions_{EMBEDDER_MODEL_NAME}"

    model = models[INDEX]
    print(f"Loading model: {EMBEDDER_MODEL_NAME}")

    print(f"\nParsing SQuAD at: {SQUAD_PATH}")
    contexts, questions = extract_squad_data(SQUAD_PATH)
    print(f"Extracted {len(contexts)} unique contexts and {len(questions)} questions.")

    question_ids = list(questions.keys())
    question_context_ids = [str(questions[qid][1]) for qid in question_ids]
    print("Length of question_context_ids:", len(question_context_ids))

    print("\nBuilding context embeddings...")
    context_ids = list(contexts.keys())
    context_texts = list(contexts.values())

    _, context_embeddings = build_and_export(
        context_ids,
        context_texts,
        model,
        name=CONTEXT_EXPORT_NAME,
        export=EXPORT,
        contextids=None,
    )

    print("\nBuilding question embeddings...")
    question_ids = list(questions.keys())
    question_texts = [q[0] for q in questions.values()]

    _, question_embeddings = build_and_export(
        question_ids,
        question_texts,
        model,
        name=QUESTION_EXPORT_NAME,
        export=EXPORT,
        contextids=question_context_ids,
    )

    persist_dir = f"chroma_db_{DATASET}_{EMBEDDER_MODEL_NAME.lower()}"
    context_collection = get_or_create_context_collection(
        collection_name=CONTEXT_EXPORT_NAME,
        ids=context_ids,
        texts=context_texts,
        embeddings=context_embeddings,
        persist_dir=persist_dir,
    )

    cross_encoder = CrossEncoder(CROSSENCODER_MODEL_NAME )
    evaluate_retrieval_with_crossencoder(
        question_embeddings=question_embeddings,
        question_context_ids=question_context_ids,
        question_texts=question_texts,
        context_collection=context_collection,
        context_texts_dict={str(cid): ctxt for cid, ctxt in zip(context_ids, context_texts)},
        crossencoder_model=cross_encoder,
        top_k=5,
        output_csv=f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv",
        question_ids=question_ids,
    )


def evaluate_liverag_crossencoder(questions_part: int):
    CONTEXT_EXPORT_NAME = f"liverag_contexts_{EMBEDDER_MODEL_NAME}"
    QUESTION_EXPORT_NAME = f"liverag_questions{questions_part}_{EMBEDDER_MODEL_NAME}"

    model = models[INDEX]
    print(f"Loading model: {EMBEDDER_MODEL_NAME}")

    contexts_df = pd.read_csv(f"{LIVERAG_PATH}/liverag_contexts.csv")
    contexts = dict(zip(contexts_df["context_id"], contexts_df["context_text"]))
    print(f"Extracted {len(contexts)} unique contexts.")

    questions_df = pd.read_csv(f"{LIVERAG_PATH}/liverag_questions{questions_part}.csv")

    if questions_part == 1:
        questions = dict(zip(
            questions_df["question_id"],
            zip(questions_df["question_text"], questions_df["answer_text"], questions_df["context_id"])
        ))
        question_context_ids = [str(questions[qid][2]) for qid in questions_df["question_id"]]
    elif questions_part == 2:
        questions = dict(zip(
            questions_df["question_id"],
            zip(questions_df["question_text"], questions_df["answer_text"], questions_df["context_ids"])
        ))
        question_context_ids = [str(questions[qid][2]) for qid in questions_df["question_id"]]
    else:
        raise ValueError(f"Unsupported questions_part: {questions_part}")

    question_ids = list(questions.keys())
    print("Length of question_context_ids:", len(question_context_ids))

    print("\nBuilding context embeddings...")
    context_ids = list(contexts.keys())
    context_texts = list(contexts.values())

    _, context_embeddings = build_and_export(
        context_ids,
        context_texts,
        model,
        name=CONTEXT_EXPORT_NAME,
        export=EXPORT,
        contextids=None,
    )

    print("\nBuilding question embeddings...")
    question_ids = list(questions.keys())
    question_texts = [q[0] for q in questions.values()]

    _, question_embeddings = build_and_export(
        question_ids,
        question_texts,
        model,
        name=QUESTION_EXPORT_NAME,
        export=EXPORT,
        contextids=question_context_ids,
    )

    persist_dir = f"chroma_db_{DATASET}_{EMBEDDER_MODEL_NAME}"
    context_collection = get_or_create_context_collection(
        collection_name=CONTEXT_EXPORT_NAME,
        ids=context_ids,
        texts=context_texts,
        embeddings=context_embeddings,
        persist_dir=persist_dir,
    )

    cross_encoder = CrossEncoder(CROSSENCODER_MODEL_NAME )
    evaluate_retrieval_with_crossencoder(
        question_embeddings=question_embeddings,
        question_context_ids=question_context_ids,
        question_texts=question_texts,
        context_collection=context_collection,
        context_texts_dict={str(cid): ctxt for cid, ctxt in zip(context_ids, context_texts)},
        crossencoder_model=cross_encoder,
        top_k=5,
        output_csv=f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{questions_part}.csv",
        question_ids=question_ids,
    )


def evaluate_liverag(questions_part: int):
    CONTEXT_EXPORT_NAME  = f"liverag_contexts_{EMBEDDER_MODEL_NAME}"
    QUESTION_EXPORT_NAME = f"liverag_questions{questions_part}_{EMBEDDER_MODEL_NAME}"
    
    model = models[INDEX]
    print(f"Loading model: {EMBEDDER_MODEL_NAME}")

    # Load contexts from LIVERAG_PATH/liverag_contexts.csv
    contexts_df = pd.read_csv(f"{LIVERAG_PATH}/liverag_contexts.csv")
    contexts = dict(zip(contexts_df["context_id"], contexts_df["context_text"]))

    print(f"Extracted {len(contexts)} unique contexts.")

    questions_df = pd.read_csv(f"{LIVERAG_PATH}/liverag_questions{questions_part}.csv")

    if questions_part == 1:
        questions = dict(zip(
            questions_df["question_id"],
            zip(questions_df["question_text"], questions_df["answer_text"], questions_df["context_id"])
        ))
        question_context_ids = [str(questions[qid][2]) for qid in questions_df["question_id"]]

    elif questions_part == 2:
        # context_ids column has two ids separated by comma
        questions = dict(zip(
            questions_df["question_id"],
            zip(questions_df["question_text"], questions_df["answer_text"], questions_df["context_ids"])
        ))
        question_context_ids = [str(questions[qid][2]) for qid in questions_df["question_id"]]

    else:
        raise ValueError(f"Unsupported questions_part: {questions_part}")

    # questions_dict: {question_id: (question_text, answer_text, context_id(s))}
    question_ids = list(questions.keys())
    print("Length of question_context_ids:", len(question_context_ids))


    print("EXPORT set to:", EXPORT)
    # --------------------------------------------------------
    # Build + export context embeddings
    # --------------------------------------------------------
    print("\nBuilding context embeddings...")

    context_ids = list(contexts.keys())
    context_texts = list(contexts.values())
    
    # Build + export
    _, context_embeddings = build_and_export(
        context_ids,
        context_texts,
        model,
        name=CONTEXT_EXPORT_NAME,
        export=EXPORT,
        contextids=None
    )

    # --------------------------------------------------------
    # Build + export question embeddings
    # --------------------------------------------------------
    print("\nBuilding question embeddings...")
    question_ids = list(questions.keys())
    question_texts = [q[0] for q in questions.values()]


    # Build + export
    _, question_embeddings = build_and_export(
        question_ids,
        question_texts,
        model,
        name=QUESTION_EXPORT_NAME,
        export=EXPORT,
        contextids=question_context_ids
    )

    if EXPORT:
        print("\nAll work complete. Embeddings exported to:")
        print(f"  {DEFAULT_EXPORT_DIR}/{CONTEXT_EXPORT_NAME}.json")
        print(f"  {DEFAULT_EXPORT_DIR}/{QUESTION_EXPORT_NAME}.json")

    persist_dir = f"chroma_db_{DATASET}_{EMBEDDER_MODEL_NAME}"

    context_collection = get_or_create_context_collection(
        collection_name=CONTEXT_EXPORT_NAME,
        ids=context_ids,
        texts=context_texts,
        embeddings=context_embeddings,
        persist_dir=persist_dir
    )
    
    # --------------------------------------------------------
    # Evaluate retrieval
    # --------------------------------------------------------

    if QUESTIONS_PART == 1:
        print("\nEvaluating retrieval for questions with 1 supporting document ...")
        evaluate_retrieval(
            question_embeddings=question_embeddings,
            question_context_ids=question_context_ids,
            context_collection=context_collection,  
            question_ids=question_ids,
            top_k=5,
            output_csv=f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{questions_part}.csv"
        )
    elif QUESTIONS_PART == 2:
        print("\nEvaluating retrieval for questions with 2 supporting documents ...")
        evaluate_retrieval_2contexts(
            question_embeddings=question_embeddings,
            question_context_ids=question_context_ids,
            context_collection=context_collection,  
            question_ids=question_ids,
            top_k=5,
            output_csv=f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{questions_part}.csv"
        )



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------



if __name__ == "__main__":
    print(f"Evaluating retrieval for {DATASET} dataset with {EMBEDDER_MODEL_NAME} ...")
    print(f"Retrieval type: {RETRIEVAL_TYPE}")
    
    if DATASET == "squad":
        if RETRIEVAL_TYPE == "crossencoder":
            print(f"Evaluating SQuAD with crossencoder {CROSSENCODER_MODEL_NAME}")
            evaluate_squad_crossencoder()
        else:
            evaluate_squad()
    if DATASET == "liverag":
        print(f"Questions part: {QUESTIONS_PART}")
        if RETRIEVAL_TYPE == "crossencoder":
            print(f"Evaluating LiveRAG with crossencoder {CROSSENCODER_MODEL_NAME}")
            evaluate_liverag_crossencoder(QUESTIONS_PART)
        else:
            evaluate_liverag(QUESTIONS_PART)        
        