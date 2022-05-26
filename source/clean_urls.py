import os
import pandas as pd

owner_id_df = pd.read_csv(os.path.join("..", "data", "owner_ids.csv"))
owner_ids = owner_id_df["owner_id"].tolist()

print(owner_ids)
print(len(owner_ids))

for owner_id in owner_ids:
    file_name = f"{owner_id}_post_attachment.csv"

    path = os.path.join("..", "data", "raw_urls", file_name)
    cleaned_path = os.path.join("..", "data", "clean_urls", file_name)

    df = pd.read_csv(path)
    df = df["Data"].str.split("\"", expand=True)  # будет в третьей колонке
    # print(df[3])
    df.rename(columns={3: "url"}, inplace=True)
    df["url"].to_csv(cleaned_path, index=False)

