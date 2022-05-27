import os
import faiss
import streamlit
from source.calculate_and_load_emb import EmbeddingCalculator

## cache?
ec = EmbeddingCalculator()

#ec.calculate_emb(os.path.join("data", "images", "all", "1.jpg"))
print(ec.calculate_emb(os.path.join("data", "images", "all", "1.jpg")))

index = faiss.read_index(os.path.join("data", "indices", "flat.index"))