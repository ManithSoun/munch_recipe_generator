import pandas as pd
import os

full_df = pd.read_csv('recipes.csv')

full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

chunk_size = 50000 

output_folder = "recipes_parts"
os.makedirs(output_folder, exist_ok=True)

for i in range(0, len(full_df), chunk_size):
    chunk = full_df.iloc[i:i+chunk_size]
    chunk.to_csv(f"{output_folder}/recipes_part_{i//chunk_size + 1}.csv", index=False)

print("Dataset shuffled and successfully split!")
