import csv
import requests

from typing import Dict
from datasets import load_dataset

from utils import extract_squad_data
from utils_liverag import extract_contexts_liverag

from config import ANSWER_GENERATION_TEMPERATURE, EMBEDDER_MODEL_NAME, RETRIEVAL_TYPE, DATASET
from config import SQUAD_PATH, LIVERAG_PATH, ANSWER_MODEL_NAME, QUESTIONS_PART, CONTEXT_REORDERED, REORDER_POSITION
from config import VERBOSE_ANSWER_GENERATION
# ===========================================================
# LLM CALL for answer generation
# ===========================================================


def call_ollama_llm(prompt: str, model_name: str=ANSWER_MODEL_NAME) -> str:
    """
    Call a local Ollama LLM and return the generated text.

    :param prompt: The prompt to send to the model
    :param model_name: The Ollama model name (e.g., 'llama3', 'mistral')
    :return: The model's response text
    """
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": ANSWER_GENERATION_TEMPERATURE,
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["response"]

import requests

def call_vllm_chat_llm(prompt: str, model_name: str=ANSWER_MODEL_NAME) -> str:
    """
    Call a vLLM chat model via OpenAI-compatible API.

    :param prompt: The user prompt
    :param model_name: The model name served by vLLM
    :return: The model's response text
    """
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": ANSWER_GENERATION_TEMPERATURE,
        "max_tokens": 512
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
    

def answer_questions_from_csv(
    csv_filename: str,
    questions: Dict[int, tuple],
    contexts: Dict[int, str],
    output_csv: str,
):
    """
    Reads retrieval results from CSV and writes LLM answers to an output CSV.

    Output CSV format (no header):
    question_id, answer
    """
    with open(csv_filename, newline="", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        counter = 0
        for row in reader:
            counter += 1
            if counter % 100 == 0:
                print(f"Processed {counter} rows...")
            if len(row) < 7:
                continue  # skip malformed rows

            question_id = int(row[0])
            retrieved_ids = [int(x) for x in row[2:7]]
            if VERBOSE_ANSWER_GENERATION:
                print(f"Processing Question ID: {question_id} with retrieved context IDs: {retrieved_ids}")
            # Get question text
            if question_id not in questions:
                continue

            question_text = questions[question_id][0]
            if VERBOSE_ANSWER_GENERATION:
                print(f"Question: {question_text}")

            # Collect retrieved context texts
            retrieved_contexts = []
            for idx, ctx_id in enumerate(retrieved_ids, start=1):
                ctx_text = contexts.get(ctx_id, "[Context not found]")
                # add the retrieved contexts in reveresed order to have the most relevant one at the end of the prompt
                retrieved_contexts.insert(0, f"Context {idx} (id={ctx_id}):\n{ctx_text}")
                # retrieved_contexts.append(
                #     f"Context {idx} (id={ctx_id}):\n{ctx_text}"
                # )

            # Build prompt
            prompt = f"""Answer the question using ONLY the information
            provided in the retrieved contexts below. I need a short, concise answer. 


            Retrieved Contexts:
            {chr(10).join(retrieved_contexts)}

            Question:
            {question_text}

            Answer:
            """
            
            # Call LLM
            # answer = call_ollama_llm(
            #     prompt=prompt,
            #     model_name=ANSWER_MODEL_NAME
            # )

            answer = call_vllm_chat_llm(
                prompt=prompt,
                model_name=ANSWER_MODEL_NAME
            )

            # Write result (no header)
            writer.writerow([question_id, answer])
        
            if VERBOSE_ANSWER_GENERATION:
                print(f"Answer: {answer}")
                print("-" * 50)



# ------------------------------------------------------------
# Main for generating LLM answers
# ------------------------------------------------------------


if __name__ == "__main__":
    print(f"Using dataset: {DATASET}")

    if DATASET == "squad":
        contexts, questions = extract_squad_data(SQUAD_PATH)
        if CONTEXT_REORDERED == False:
            INPUT_FILE = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv"
            OUTPUT_FILE = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_newprompt.csv"
        else:
            INPUT_FILE = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{REORDER_POSITION}.csv"
            OUTPUT_FILE = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{REORDER_POSITION}_newprompt.csv"
    elif DATASET == "liverag":  
        ds = load_dataset("LiveRAG/Benchmark")
        contexts, questions1, questions2 = extract_contexts_liverag(ds, split="train")
        if QUESTIONS_PART == 1:
            questions = questions1
        elif QUESTIONS_PART == 2:
            questions = questions2
        else:
            raise ValueError(f"Unsupported questions_part: {QUESTIONS_PART}")
        if CONTEXT_REORDERED == False:
            INPUT_FILE = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}.csv"
            OUTPUT_FILE = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_newprompt.csv"
        else:
            INPUT_FILE = f"data/retriever/{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_{REORDER_POSITION}.csv"
            OUTPUT_FILE = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_{REORDER_POSITION}_newprompt.csv"

    print(f"Extracted {len(contexts)} unique contexts and {len(questions)} questions.")

    print("Context reordering settings:" )
    print(f"CONTEXT_REORDERED: {CONTEXT_REORDERED}")
    if CONTEXT_REORDERED:
        print(f"REORDER_POSITION: {REORDER_POSITION}")
    print(f"REORDER_POSITION: {REORDER_POSITION}")
    print(f"Input CSV: {INPUT_FILE}")
    print(f"Output CSV: {OUTPUT_FILE}")

    print(f"\nAnswer questions from CSV: {INPUT_FILE}\n")
    answer_questions_from_csv(
        csv_filename=INPUT_FILE,
        questions=questions,
        contexts=contexts,
        output_csv=OUTPUT_FILE
    )
