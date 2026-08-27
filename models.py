import numpy as np

from sentence_transformers import SentenceTransformer
from config import BATCH_SIZE

class SentenceTransformerEmbedder:
    def __init__(self, model_name="BAAI/bge-m3", normalize_embeddings=True):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.normalize_embeddings = normalize_embeddings
        # Access tokenizer's model_max_length
        tokenizer_max = self.model.tokenizer.model_max_length


    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=None):
        if isinstance(texts, str):
            texts = [texts]

        normalize = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            batch_size=BATCH_SIZE, 
        )
        return embeddings
    

