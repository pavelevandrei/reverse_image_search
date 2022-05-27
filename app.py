import numpy as np
import os
import faiss
import streamlit
from source.calculate_and_load_emb import EmbeddingCalculator

## cache?
TOPN = 6

ec = EmbeddingCalculator()

def load_index():
    index = faiss.read_index(os.path.join("data", "indices", "flat.index"))
    return index

index=load_index()

search_vector = np.array(ec.calculate_emb(os.path.join("data", "images", "all", "1.jpg")), dtype='float32').reshape(1,-1)

print(search_vector.shape)



D, I = index.search(search_vector, TOPN)
print(I)