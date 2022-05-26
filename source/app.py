import os
import pickle
from time import sleep
import urllib.request

import pandas as pd
from towhee import pipeline

embedding_pipeline = pipeline('image-embedding')
# embedding = embedding_pipeline('https://docs.towhee.io/img/logo.png')
# print(type(embedding_pipeline))
# print(embedding)

owner_id_df = pd.read_csv(os.path.join("..", "data", "owner_ids.csv"))
owner_ids = owner_id_df["owner_id"].tolist()

url_list = []

for owner_id in owner_ids:
    file_name = f"{owner_id}_post_attachment.csv"

    url_path = os.path.join("..", "data", "clean_urls", file_name)
    df = pd.read_csv(url_path)
    urls = df["url"].tolist()
    url_list.extend(urls)

print(len(url_list))

embedding_list = []
for i, url in enumerate(urls[:]):
    print(f"Обрабатывает {i} изображение")
    urllib.request.urlretrieve(url, os.path.join("..", "data", "images", "all", f"{i}.jpg"))
    embedding = embedding_pipeline(url)
    embedding_list.append(embedding)
    sleep(0.01)


emb_path = os.path.join("..", "data", "embeddings", f"{owner_id}.pickle")
with open(emb_path, 'wb') as file:
    pickle.dump(embedding_list, file)

#print(embedding_list)