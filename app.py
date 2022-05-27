import numpy as np
import os
import faiss
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from source.calculate_and_load_emb import EmbeddingCalculator

## cache?
TOPN = 9

ec = EmbeddingCalculator()

@st.cache
def load_index():
    index = faiss.read_index(os.path.join("data", "indices", "flat.index"))
    return index
index=load_index()

st.title(f"Поиск похожих изображений для модерации сетевых сообществ")
st.markdown(f"""
Поиск похожих изображений в сообществах ВКонтакте **"Барахолки"** и **"Объявления"** .

Используемый подход:
* Собираем базу изображений
* Присваиваем каждому изображению идентификатор
* Рассчитываем векторное представление (эмбеддинг), используемая модель **resnet50**
* Полученные векторные представления индексируем в эффективной структуре для поиска с помощью библиотеки **faiss**
* Пользователь может загрузить изображение и найти похожие на него изображения в сообществах-барахолках
*** 
""")

images_count = len(os.listdir(os.path.join("data", "images", "all")))

image_number = int(st.number_input(label="Введите номер изображения", value=0, min_value=0, max_value=images_count-1, step=1))

image = mpimg.imread(os.path.join("data", "images", "all", f"{image_number}.jpg"))

st.image(image)

search_vector = np.array(ec.calculate_emb(os.path.join("data", "images", "all", f"{image_number}.jpg")), dtype='float32').reshape(1,-1)
norm = np.linalg.norm(search_vector)
search_vector = search_vector / norm

st.write(search_vector)
st.caption("Эмбеддинг изображения")


print(search_vector.shape)


D, I = index.search(search_vector, TOPN)
print(D)
print(I)

st.markdown(f"""
***
Похожие изображения

""")
st.write(I)
st.write(D)
col1, col2, col3 = st.columns(3)

image1 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][0]}.jpg"))

image2 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][1]}.jpg"))
image3 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][2]}.jpg"))
col1.image(image1)
col1.caption(f"Идентификатор изображения - {I[0][0]}, расстояние - {round(D[0][0])}")
col2.image(image2)
col2.caption(f"Идентификатор изображения - {I[0][1]}, расстояние - {round(D[0][1])}")
col3.image(image3)
col3.caption(f"Идентификатор изображения - {I[0][2]}, расстояние - {round(D[0][2])}")

col1, col2, col3 = st.columns(3)

image1 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][3]}.jpg"))
image2 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][4]}.jpg"))
image3 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][5]}.jpg"))
col1.image(image1)
col1.caption(f"Идентификатор изображения - {I[0][3]}, расстояние - {round(D[0][3])}")
col2.image(image2)
col2.caption(f"Идентификатор изображения - {I[0][4]}, расстояние - {round(D[0][4])}")
col3.image(image3)
col3.caption(f"Идентификатор изображения - {I[0][5]}, расстояние - {round(D[0][5])}")

col1, col2, col3 = st.columns(3)

image1 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][6]}.jpg"))
image2 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][7]}.jpg"))
image3 = mpimg.imread(os.path.join("data", "images", "all", f"{I[0][8]}.jpg"))
col1.image(image1)
col1.caption(f"Идентификатор изображения - {I[0][6]}, расстояние - {round(D[0][6])}")
col2.image(image2)
col2.caption(f"Идентификатор изображения - {I[0][7]}, расстояние - {round(D[0][7])}")
col3.image(image3)
col3.caption(f"Идентификатор изображения - {I[0][8]}, расстояние - {round(D[0][8])}")