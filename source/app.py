import os
import pickle
from time import sleep
import pandas as pd
from towhee import pipeline

embedding_pipeline = pipeline('image-embedding')
embedding = embedding_pipeline('https://docs.towhee.io/img/logo.png')

print(type(embedding_pipeline))
print(embedding)

file_name = "78815338_post_attachment.csv"
emb_file = file_name.split("_")[0]
url_path = os.path.join("..", "data", "clean_urls", file_name)
df = pd.read_csv(url_path)
urls = df["url"].tolist()
#print(urls)
embedding_list = []
for i, url in enumerate(urls[:5]):
    print(f"Обрабатывает {i} изображение")
    
    embedding = embedding_pipeline(url)
    embedding_list.append(embedding)
    sleep(0.4)

emb_path = os.path.join("..", "data", "embeddings", emb_file)
with open(emb_path, 'wb') as file:
    pickle.dump(embedding_list, file)

print(embedding_list)