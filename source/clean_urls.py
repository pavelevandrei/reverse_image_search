import os
import pandas as pd

file_name = "78815338_post_attachment.csv"

path = os.path.join("..", "data", "raw_urls", file_name)
cleaned_path = os.path.join("..", "data", "clean_urls", file_name)

df = pd.read_csv(path)
df = df["Data"].str.split("\"", expand=True)  # будет в третьей колонке
#print(df[3])
df.rename(columns={3: "url"}, inplace=True)
df["url"].to_csv(cleaned_path, index=False)
