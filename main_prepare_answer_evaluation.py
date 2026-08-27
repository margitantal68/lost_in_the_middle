import csv
from config import  DATASET, RETRIEVAL_TYPE, EMBEDDER_MODEL_NAME, QUESTIONS_PART, CONTEXT_REORDERED, REORDER_POSITION
from config import CONTEXT_REORDERED, REORDER_POSITION, fold, IS_FOLD

def create_input_files_for_answer_evaluation_squad_fold(fold: str) -> None:
    """
    Create input CSV file containing only questions present in INPUT_FILE_3.
    """

    INPUT_FILE_1 = "data/answers/squad_questions.csv"
    INPUT_FILE_2 = "data/answers/answers_squad.csv"
    INPUT_FILE_3 = f"data/answers/answers_{fold}.csv"

    OUTPUT_FILE = f"data/answers/input_answers_{fold}.csv"

    # Load questions: {id: question_text}
    questions = {}
    with open(INPUT_FILE_1, newline="", encoding="utf-8") as f1:
        reader1 = csv.reader(f1)
        for row in reader1:
            question_id, question_text = row
            questions[question_id] = question_text

    # Load ground truth answers: {id: answer}
    answers = {}
    with open(INPUT_FILE_2, newline="", encoding="utf-8") as f2:
        reader2 = csv.reader(f2)
        for row in reader2:
            question_id, answer = row
            answers[question_id] = answer

    # Process only IDs from INPUT_FILE_3
    with open(INPUT_FILE_3, newline="", encoding="utf-8") as f3, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f4:

        reader3 = csv.reader(f3)
        writer = csv.writer(f4)

        writer.writerow(["question_id", "question_text", "answer", "llm_answer"])

        for row in reader3:
            question_id, llm_answer = row

            # Get corresponding data
            question_text = questions.get(question_id, "")
            answer = answers.get(question_id, "")

            writer.writerow([question_id, question_text, answer, llm_answer])

def create_input_files_for_answer_evaluation_squad()->None:
    """
    Create input CSV files for LLM answer evaluation from SQuAD data.
    """

    INPUT_FILE_1 = "data/answers/squad_questions.csv"
    INPUT_FILE_2 = f"data/answers/answers_squad.csv"
    if CONTEXT_REORDERED == True:
        INPUT_FILE_3 = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{REORDER_POSITION}_newprompt.csv"
    else:
        INPUT_FILE_3 = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_newprompt.csv"

    OUTPUT_FILE = f"data/answers/input_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}.csv" 

    # INPUT_FILE_1: question_id, question_text
    # INPUT_FILE_2: question_id, answer
    # INPUT_FILE_3
    # OUTPUT_FILE: question_id, question_text, answer, llm_answer

    with open(INPUT_FILE_1, newline="", encoding="utf-8") as f1, \
         open(INPUT_FILE_2, newline="", encoding="utf-8") as f2, \
         open(INPUT_FILE_3, newline="", encoding="utf-8") as f3, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f4:

        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        reader3 = csv.reader(f3)

        writer = csv.writer(f4)
        writer.writerow(["question_id", "question_text", "answer", "llm_answer"])

        for row1, row2, row3 in zip(reader1, reader2, reader3):
            question_id = row1[0]
            question_text = row1[1]
            answer = row2[1]
            llm_answer = row3[1]

            writer.writerow([question_id, question_text, answer, llm_answer])
  

def create_input_files_for_answer_evaluation_liverag()->None:
    """
    Create input CSV files for LLM answer evaluation from SQuAD data.
    """


    INPUT_FILE_1 = "data/liverag/liverag_questions1.csv"
    if CONTEXT_REORDERED == True:
        INPUT_FILE_3 = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_{REORDER_POSITION}_newprompt.csv"
    else:
        INPUT_FILE_3 = f"data/answers/answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}_newprompt.csv"

    OUTPUT_FILE = f"data/answers/input_answers_{DATASET}_{RETRIEVAL_TYPE}_{EMBEDDER_MODEL_NAME}_{QUESTIONS_PART}.csv" 

    with open(INPUT_FILE_1, newline="", encoding="utf-8" ) as f1, \
         open(INPUT_FILE_3, newline="", encoding="utf-8") as f3, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f4:

        reader1 = csv.reader(f1)
        # 1. Skip the header in f1 (the file that HAS a header)
        next(reader1)

        reader3 = csv.reader(f3)

        writer = csv.writer(f4)
        writer.writerow(["question_id", "question_text", "answer", "llm_answer"])

        for row1, row3 in zip(reader1, reader3):
            question_id = row1[0]
            question_text = row1[1]
            answer = row1[2]
            llm_answer = row3[1]
            writer.writerow([question_id, question_text, answer, llm_answer])
  
import csv

def create_input_files_for_answer_evaluation_liverag_fold(fold: str) -> None:
    """
    Create input CSV file for LLM answer evaluation by joining
    questions with LLM answers on question_id.
    """

    INPUT_FILE_1 = "data/liverag/liverag_questions1.csv"
    INPUT_FILE_3 = f"data/answers/answers_{fold}.csv"
    OUTPUT_FILE = f"data/answers/input_answers_{fold}.csv"

    # Step 1: Load INPUT_FILE_3 into a dictionary (question_id -> llm_answer)
    llm_answers = {}
    with open(INPUT_FILE_3, newline="", encoding="utf-8") as f3:
        reader3 = csv.reader(f3)
        for row in reader3:
            question_id = row[0]
            llm_answer = row[1]
            llm_answers[question_id] = llm_answer

    # Step 2: Iterate through INPUT_FILE_1 and match
    with open(INPUT_FILE_1, newline="", encoding="utf-8") as f1, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f4:

        reader1 = csv.reader(f1)
        writer = csv.writer(f4)

        # Skip header in INPUT_FILE_1
        next(reader1)

        # Write output header
        writer.writerow(["question_id", "question_text", "answer", "llm_answer"])

        for row1 in reader1:
            question_id = row1[0]

            # Only keep rows that exist in INPUT_FILE_3
            if question_id in llm_answers:
                question_text = row1[1]
                answer = row1[2]
                llm_answer = llm_answers[question_id]

                writer.writerow([question_id, question_text, answer, llm_answer])


if __name__ == "__main__":
    if IS_FOLD:
        print(f"fold-based evaluation is enabled: {fold}")
        if DATASET == "squad":
            create_input_files_for_answer_evaluation_squad_fold(fold)
        elif DATASET == "liverag":
            create_input_files_for_answer_evaluation_liverag_fold(fold)
    else:
        if DATASET == "squad":
            create_input_files_for_answer_evaluation_squad()
        elif DATASET == "liverag":
            create_input_files_for_answer_evaluation_liverag()
    
