import csv
import os
import requests
from typing import Dict
from dotenv import load_dotenv

from config import DATASET, EMBEDDER_MODEL_NAME, RETRIEVAL_TYPE, ANSWER_GENERATION_TEMPERATURE, JUDGE_MODEL_NAME, QUESTIONS_PART
from config import CONTEXT_REORDERED, REORDER_POSITION, START_QUESTION_ID, ANSWER_MODEL_NAME, ANSWER_MODEL_CODE
from config import fold, IS_FOLD
load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
VLLM_URL = "http://localhost:8000/v1/chat/completions"

JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator for question-answering tasks.

Your job is to evaluate whether the LLM answer correctly answers the question,
based on the provided ground truth answer.

Rules:
- Focus on factual correctness and completeness.
- Ignore differences in wording or style.
- Do not reward unsupported extra information.
- Do not penalize correct paraphrasing.

Scoring rubric:
5 = Fully correct and equivalent to the ground truth
4 = Mostly correct, very minor omission or imprecision
3 = Partially correct, missing key information
2 = Mostly incorrect, only small correct elements
1 = Completely incorrect or unrelated

Question:
{question}

Ground truth answer:
{answer}

LLM answer:
{llm_answer}

Output exactly two lines:

SCORE: <integer from 1 to 5>
EXPLANATION: <brief explanation>
"""


def parse_lines(text: str):
    score = None
    explanation = ""

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip())
            except ValueError:
                score = None

        elif line.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    return {
        "score": score,
        "explanation": explanation if explanation else None
    }



def call_ollama(prompt: str, ollama_url: str, judge_model: str=JUDGE_MODEL_NAME, temperature: float=ANSWER_GENERATION_TEMPERATURE) -> Dict:
    """Call Ollama and return parsed JSON response."""
    payload = {
        "model": judge_model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()

    raw_text = response.json()["response"].strip()

    parsed = parse_lines(raw_text)

    # Fallback if parsing failed
    if parsed["score"] is None:
        return {
            "score": None,
            "explanation": f"Failed to parse judge output. Raw output: {raw_text}"
        }

    return parsed



def call_vllm(
    prompt: str,
    model_url: str = VLLM_URL,
    model_name: str = JUDGE_MODEL_NAME,
    temperature: float = 0
) -> Dict:
    """Call a vLLM-served chat model and return parsed JSON response."""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(model_url, json=payload, headers=headers)
    response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"].strip()

    parsed = parse_lines(raw_text)

    # Fallback if parsing failed
    if parsed["score"] is None:
        return {
            "score": None,
            "explanation": f"Failed to parse judge output. Raw output: {raw_text}"
        }

    return parsed


def evaluate_row(question: str, answer: str, llm_answer: str, model_url: str, judge_model: str, temperature: float) -> Dict:
    """Evaluate a single QA pair using LLM-as-a-judge."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        llm_answer=llm_answer
    )
    # return call_ollama(prompt, ollama_url, judge_model, temperature)
    return call_vllm(prompt, model_url, model_name=judge_model, temperature=temperature)



def evaluate_csv(
    input_csv_path: str,
    output_csv_path: str,
    model_url: str,
    judge_model: str,
    temperature: float,
):
    """Evaluate CSV rows and optionally resume from START_QUESTION_ID."""
    append_mode = START_QUESTION_ID > 0
    output_exists = os.path.exists(output_csv_path)

    with open(input_csv_path, newline="", encoding="utf-8") as infile, \
         open(output_csv_path, "a" if append_mode else "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = [
            "question_id",
            "question",
            "answer",
            "llm_answer",
            "score",
            "explanation"
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if not append_mode or not output_exists or os.path.getsize(output_csv_path) == 0:
            writer.writeheader()
        counter = 0
        for row in reader:
            counter = counter + 1
            if counter % 100 == 0:
                print(f"Processed {counter} rows...")
            question_id = int(row["question_id"])
            if question_id < START_QUESTION_ID:
                continue

            question = row["question_text"]
            answer = row["answer"]
            llm_answer = row["llm_answer"]

            result = evaluate_row(question, answer, llm_answer, model_url, judge_model, temperature)
            # print(question_id, result.get("score"))
            writer.writerow({
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "llm_answer": llm_answer,
                "score": result.get("score"),
                "explanation": result.get("explanation"),
            })



# ------------------------------------------------------------
# Main for evaluating LLM answers
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using dataset: {DATASET}")
    print(f"Using embedder model: {EMBEDDER_MODEL_NAME}")
    print(f"Using retrieval type: {RETRIEVAL_TYPE}")
    print(f"Using context reordering: {CONTEXT_REORDERED}")
    if CONTEXT_REORDERED:
        print(f"Using reorder position: {REORDER_POSITION}")

    # Fold-based evaluation
    if IS_FOLD:
        print(f"fold-based evaluation is enabled: {fold}")
        INPUT_FILE = f"data/answers/input_answers_{fold}.csv" 
        OUTPUT_FILE = f"data/answers/evaluated_output_answers_{fold}.csv"

    else:
        if DATASET == "squad":
            INPUT_FILE = f"data/answers/input_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv" 
        else:
            INPUT_FILE = f"data/answers/input_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}.csv"
        if CONTEXT_REORDERED == True:
            OUTPUT_FILE = f"data/answers/_evaluated_output_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{REORDER_POSITION}_{ANSWER_MODEL_CODE}.csv"
        else:
            OUTPUT_FILE = f"data/answers/_evaluated_output_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{ANSWER_MODEL_CODE}.csv"
    print(f"Input CSV: {INPUT_FILE}")
    print(f"Output CSV: {OUTPUT_FILE}")
    
    if __name__ == "__main__":
        evaluate_csv(
            input_csv_path=INPUT_FILE,
            output_csv_path=OUTPUT_FILE,
            model_url=VLLM_URL,
            judge_model=JUDGE_MODEL_NAME,
            temperature=0,
        )
