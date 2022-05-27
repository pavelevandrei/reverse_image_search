import os
import pickle
import numpy as np
import faiss


TOPN = 6
FAISS_PARAMETERS_STRING = "IVF120,SQ8" # тип индекса

# загружаем массив эмбеддингов из файла
emb_path = os.path.join("..", "data", "embeddings", f"all.pickle")
with open(emb_path, 'rb') as file:
    embedding_list = pickle.load(file)
emb_array = np.array(embedding_list, np.float32)

# индентификаторы - порядок файла в папке
ids = np.array(range(0, emb_array.shape[0]))
ids = np.asarray(ids.astype('int64'))

# создаём индекс
index = faiss.index_factory(emb_array.shape[1], FAISS_PARAMETERS_STRING, faiss.METRIC_INNER_PRODUCT)
index.train(emb_array)
index.add_with_ids(emb_array, ids)

# сохраняем индекс
faiss.write_index(index, os.path.join("..", "data", "indices", "ivf.index"))

# проверяем работоспособность
print(index.ntotal)
print(emb_array[0:1])
D, I = index.search(emb_array[0:1], TOPN)
print(I)
print(D)

