import os
import pickle

import numpy as np
import faiss

owner_id = "78815338"
emb_path = os.path.join("..", "data", "embeddings", f"{owner_id}.pickle")
with open(emb_path, 'rb') as file:
    embedding_list = pickle.load(file)
emb_array = np.array(embedding_list, np.float32)
norm = np.linalg.norm(emb_array)
emb_array = emb_array / norm # нормализация

# index = faiss.IndexFlatL2((emb_array.shape[1])
