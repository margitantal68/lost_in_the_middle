import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

from utils import save_embeddings_json
from config import BATCH_SIZE, DEFAULT_EXPORT_DIR



def _ensure_export_dir(export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)


# RTE Embeddings Builders and Exporters
def build_embeddings_for_texts(ids: List[int], texts: List[str], model, batch_size: int = BATCH_SIZE) -> Tuple[List[str], np.ndarray]:
    """Encode texts using the given model and return string ids and numpy embeddings.

    Note: model must expose `.encode(...)` returning numpy arrays when `convert_to_numpy=True`.
    """
    # Ensure texts is a list
    if not isinstance(texts, list):
        texts = list(texts)

    # Convert ids to strings to be consistent with Chroma ids
    str_ids = [str(i) for i in ids]

    print(f"Encoding {len(texts)} texts with batch_size={batch_size}...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    print("Done encoding.")
    embeddings = np.array(embeddings)

    return str_ids, embeddings


def export_embeddings_json(ids: List[str], texts: List[str], embeddings: np.ndarray, out_path: Path, contextids: List[str]=None) -> None:
    """Export list of {id,text,embedding} items to JSON file.

    The `embeddings` arg is a 2D numpy array of shape (n, dim).
    """
    assert embeddings.shape[0] == len(ids) == len(texts), "Length mismatch"
    items = []
    for i, _id in enumerate(ids):
        if contextids:
            items.append({
                "id": _id,
                "text": texts[i],
                "embedding": embeddings[i].tolist(),
                "context_id": contextids[i],
            })
        else:
            items.append({
                "id": _id,
                "text": texts[i],
                "embedding": embeddings[i].tolist(),
            })

    save_embeddings_json(items, str(out_path))
    print(f"Exported embeddings to {out_path}")


def build_and_export(ids: List[int], 
                     texts: List[str], 
                     model, 
                     name: str, 
                     export: bool = False, 
                     contextids=None ) -> Tuple[List[str], np.ndarray]:
    """Convenience: build embeddings and export to `<export_dir>/<name>.json`.

    If `in_export_dir` is None, embeddings are not exported.

    Returns (str_ids, embeddings)
    """
    if not isinstance(texts, list):
        texts = list(texts)

    str_ids, embeddings = build_embeddings_for_texts(ids, texts, model)

    if export:
        export_dir = Path(DEFAULT_EXPORT_DIR)
        _ensure_export_dir(export_dir)
        out_path = export_dir / f"{name}.json"
        export_embeddings_json(str_ids, texts, embeddings, out_path, contextids=contextids)

    return str_ids, embeddings