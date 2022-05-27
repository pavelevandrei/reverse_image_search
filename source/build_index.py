import os
import pickle

import numpy as np
import faiss

TOPN = 6
# owner_id = "78815338"
emb_path = os.path.join("..", "data", "embeddings", f"all.pickle")
with open(emb_path, 'rb') as file:
    embedding_list = pickle.load(file)
emb_array = np.array(embedding_list, np.float32)
norm = np.linalg.norm(emb_array)
emb_array = emb_array / norm # нормализация

ids = np.array(range(0, emb_array.shape[0]))
ids = np.asarray(ids.astype('int64'))


index = index = faiss.IndexIDMap(faiss.IndexFlatL2(emb_array.shape[1])) # IndexFlatIP для косинусного расстояния
index.add_with_ids(emb_array, ids)
print(index.ntotal)
D, I = index.search(emb_array[5333:5338], TOPN)
print(I)
print(D)