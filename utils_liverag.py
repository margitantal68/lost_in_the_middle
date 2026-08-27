import numpy as np
import pandas as pd

from datasets import load_dataset
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer





# write a function to extract the content field from a string having the following format: 
# "{'content': <content text>, 'doc_id': <doc_id>}" and return the content text. 
# If the string does not have this format, return the original string.

def extract_content(content_str: str) -> str:
	try:
		content_dict = eval(content_str)
		return content_dict['content']
	except (SyntaxError, ValueError):
		return content_str.strip()

def extract_contexts_liverag(
	dataset: Any,
	split: str = "train",
	field: str = "Supporting_Documents",
	question_field: str = "Question",
	answer_field: str = "Answer",
) -> Tuple[Dict[int, str], Dict[int, Tuple[str, str, int, float, float, float, float]], Dict[int, Tuple[str, str, List[int], float, float, float, float]]]:
	"""Extract supporting documents and map questions by support count with metadata fields.

	Args:
		dataset: a `DatasetDict` or `Dataset` returned by `datasets.load_dataset`.
		split: which split to read when `dataset` is a `DatasetDict` (default: 'train').
		field: name of the field that contains supporting documents (default: 'Supporting_Documents').
		question_field: name of the field that contains question text (default: 'Question').
		answer_field: name of the field that contains answer text (default: 'Answer').

	Returns:
		contexts: mapping from context_id -> document text (all unique documents).
		questions1: mapping from question_id -> (question_text, answer, context_id, ACS, ACS_Std, IRT-diff, IRT-disc) for questions with exactly 1 supporting doc.
		questions2: mapping from question_id -> (question_text, answer, [context_id1, context_id2], ACS, ACS_Std, IRT-diff, IRT-disc) for questions with exactly 2 supporting docs.

	Notes:
		- If a supporting-document entry is a list, each element is treated as a separate document.
		- If it's a string, it's treated as a single document.
		- Documents with identical text across examples are deduplicated and share the same id.
		- Extracts metadata fields: ACS, ACS_Std, IRT-diff, IRT-disc from the dataset.
	"""
	# If a DatasetDict was passed, select the requested split
	ds_split = dataset
	try:
		# datasets.DatasetDict supports dict-like access
		if hasattr(dataset, "keys") and split in dataset:
			ds_split = dataset[split]
	except Exception:
		ds_split = dataset

	doc_text_to_id: Dict[str, int] = {}
	context_dict: Dict[int, str] = {}
	questions1: Dict[int, Tuple[str, str, int, float, float, float, float]] = {}
	questions2: Dict[int, Tuple[str, str, List[int], float, float, float, float]] = {}
	next_context_id = 0
	question_id = 0

	for item in ds_split:
		# Extract question text
		question_text = item.get(question_field, None) if isinstance(item, dict) else getattr(item, question_field, None)
		if question_text is None:
			question_text = ""
		else:
			question_text = str(question_text).strip()

		# Extract answer text
		answer_text = item.get(answer_field, None) if isinstance(item, dict) else getattr(item, answer_field, None)
		if answer_text is None:
			answer_text = ""
		else:
			answer_text = str(answer_text).strip()

		# Extract metadata fields
		acs = item.get("ACS [-2 : 1]", None) if isinstance(item, dict) else getattr(item, "ACS [-2 : 1]", None)
		acs = float(acs) if acs is not None else 0.0

		acs_std = item.get("ACS_Std", None) if isinstance(item, dict) else getattr(item, "ACS_Std", None)
		acs_std = float(acs_std) if acs_std is not None else 0.0

		irt_diff = item.get("IRT-diff [-6 : 6]", None) if isinstance(item, dict) else getattr(item, "IRT-diff [-6 : 6]", None)
		irt_diff = float(irt_diff) if irt_diff is not None else 0.0

		irt_disc = item.get("IRT-disc [-0.6 : 1.4]", None) if isinstance(item, dict) else getattr(item, "IRT-disc [-0.6 : 1.4]", None)
		irt_disc = float(irt_disc) if irt_disc is not None else 0.0

		# Extract supporting documents
		raw = item.get(field, None) if isinstance(item, dict) else getattr(item, field, None)
		docs: List[str] = []
		if raw is None:
			docs = []
		elif isinstance(raw, list):
			# assume list of strings (or convertible to str)
			docs = [str(d).strip() for d in raw if d is not None and str(d).strip()]
		else:
			# single string or other scalar
			s = str(raw).strip()
			docs = [s] if s else []

		# Map documents to context ids and collect unique context text
		context_ids_for_question: List[int] = []
		context_texts_for_question: List[str] = []

		for d in docs:
			
			if d not in doc_text_to_id:
				d = extract_content(d)
				print(f"New context found (ID {next_context_id}): {d[:100]}...")
				doc_text_to_id[d] = next_context_id
				
				context_dict[next_context_id] = d
				next_context_id += 1
			context_id = doc_text_to_id[d]
			context_ids_for_question.append(context_id)
			context_texts_for_question.append(d)

		# Categorize questions by number of supporting documents
		num_docs = len(context_ids_for_question)
		if num_docs == 1:
			questions1[question_id] = (question_text, answer_text, context_ids_for_question[0], acs, acs_std, irt_diff, irt_disc)
		elif num_docs == 2:
			questions2[question_id] = (question_text, answer_text, context_ids_for_question, acs, acs_std, irt_diff, irt_disc)

		question_id += 1

	return context_dict, questions1, questions2



def statistics(contexts: Optional[Dict[int, str]] = None, *,
			   tokenizer_model: str = "BAAI/bge-m3") -> None:

	# Initialize tokenizer and model max length
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
	MAX_TOKENS = tokenizer.model_max_length
	print(f"Using tokenizer from model: {tokenizer_model}")
	print(f"Model's tokenizer max sequence length (MAX_TOKENS): {MAX_TOKENS}\n")


	# Compute token counts for each context
	token_counts = []
	contexts_exceeding_MAX_TOKENS = []

	for context_id, context_text in contexts.items():
		token_count = len(tokenizer.encode(context_text))
		token_counts.append(token_count)

		if token_count > MAX_TOKENS:
			contexts_exceeding_MAX_TOKENS.append((context_id, token_count))

	if not token_counts:
		print("No contexts to analyze.")
		return

	token_counts_array = np.array(token_counts)

	# Compute statistics
	min_tokens = int(np.min(token_counts_array))
	max_tokens = int(np.max(token_counts_array))
	avg_tokens = float(np.mean(token_counts_array))
	std_tokens = float(np.std(token_counts_array))

	# Print statistics
	print("=" * 50)
	print("CONTEXT TOKEN COUNT STATISTICS")
	print("=" * 50)
	print(f"Number of contexts: {len(contexts)}")
	print(f"Minimum tokens:     {min_tokens}")
	print(f"Maximum tokens:     {max_tokens}")
	print(f"Average tokens:     {avg_tokens:.2f}")
	print(f"Std deviation:      {std_tokens:.2f}")
	print("=" * 50)

	# Print contexts exceeding MAX_TOKENS tokens
	print(f"\nContexts exceeding {MAX_TOKENS} tokens: {len(contexts_exceeding_MAX_TOKENS)}")
	print("=" * 50)




if __name__ == "__main__":
	# Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("LiveRAG/Benchmark")
	
    contexts, questions1, questions2 = extract_contexts_liverag(ds, split="train")
	
    print(f"Extracted {len(contexts)} unique supporting documents from train split")
    print(f"Questions with exactly 1 supporting document: {len(questions1)}")
    print(f"Questions with exactly 2 supporting documents: {len(questions2)}")

	# Export contexts to a CSV file into the data/liverag folder
    contexts_df = pd.DataFrame({"context_id": list(contexts.keys()), "context_text": list(contexts.values())})
    contexts_df.to_csv("data/liverag/liverag_contexts.csv", index=False)
	# Export questions with 1 supporting document to a CSV file
    questions1_df = pd.DataFrame({	"question_id": list(questions1.keys()),
									"question_text": [q[0] for q in questions1.values()],
									"answer_text": [q[1] for q in questions1.values()],
									"context_id": [q[2] for q in questions1.values()],
									"ACS": [q[3] for q in questions1.values()],
									"ACS_Std": [q[4] for q in questions1.values()],
									"IRT-diff": [q[5] for q in questions1.values()],
									"IRT-disc": [q[6] for q in questions1.values()]})
    questions1_df.to_csv("data/liverag/liverag_questions1.csv", index=False)

	# Export questions with 2 supporting documents to a CSV file
    questions2_df = pd.DataFrame({	"question_id": list(questions2.keys()),
									"question_text": [q[0] for q in questions2.values()],
									"answer_text": [q[1] for q in questions2.values()],
									"context_ids": [",".join(map(str, q[2])) for q in questions2.values()],
									"ACS": [q[3] for q in questions2.values()],
									"ACS_Std": [q[4] for q in questions2.values()],
									"IRT-diff": [q[5] for q in questions2.values()],
									"IRT-disc": [q[6] for q in questions2.values()]})
    questions2_df.to_csv("data/liverag/liverag_questions2.csv", index=False)

    print("*" * 50)
    statistics(contexts)
    print("*" * 50)
    
    # Print first 5 contexts for debugging
    print("First 5 contexts:")
    for ctx_id, ctx_text in list(contexts.items())[:5]:
        print(f"Context ID {ctx_id}: {ctx_text[:100]}...")
    
    print("\n" + "*" * 50)
    print("First 3 questions with 1 supporting document:")
    for q_id, q_data in list(questions1.items())[:3]:
        print(f"Question ID {q_id}: {q_data[0][:100]}...")
        print(f"  Answer: {q_data[1][:100]}...")
        print(f"  Context ID: {q_data[2]}")
        print(f"  ACS: {q_data[3]:.4f}, ACS_Std: {q_data[4]:.4f}")
        print(f"  IRT-diff: {q_data[5]:.4f}, IRT-disc: {q_data[6]:.4f}")
    
    print("\n" + "*" * 50)
    print("First 3 questions with 2 supporting documents:")
    for q_id, q_data in list(questions2.items())[:3]:
        print(f"Question ID {q_id}: {q_data[0][:100]}...")
        print(f"  Answer: {q_data[1][:100]}...")
        print(f"  Context IDs: {q_data[2]}")
        print(f"  ACS: {q_data[3]:.4f}, ACS_Std: {q_data[4]:.4f}")
        print(f"  IRT-diff: {q_data[5]:.4f}, IRT-disc: {q_data[6]:.4f}")
        for ctx_id in q_data[2]:
            print(f"    Context {ctx_id}: {contexts[ctx_id][:100]}...")
        
    print("*" * 50)
   