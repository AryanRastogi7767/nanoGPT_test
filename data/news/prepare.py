import os
import pickle
import numpy as np
from datasets import load_dataset

# Load a news dataset (CC-News for diverse articles)
dataset = load_dataset("cc_news", split="train")
text_data = "\n".join(dataset["text"])  # Combine all articles into a single text

# Define paths
data_dir = "data/news_articles"
os.makedirs(data_dir, exist_ok=True)
input_file_path = os.path.join(data_dir, "input.txt")

# Save dataset to input.txt
with open(input_file_path, "w", encoding="utf-8") as f:
    f.write(text_data)

print(f"Dataset saved at {input_file_path}. Length: {len(text_data):,} characters.")

# Get unique characters
chars = sorted(list(set(text_data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi.get(c, 0) for c in s]  # Handle unknown chars


def decode(l):
    return "".join([itos.get(i, "?") for i in l])  # Handle unknown indices


# Split data into train and validation sets
n = len(text_data)
train_data = text_data[: int(n * 0.9)]
val_data = text_data[int(n * 0.9) :]

# Encode to integers
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

# Save binary files
train_ids.tofile(os.path.join(data_dir, "train.bin"))
val_ids.tofile(os.path.join(data_dir, "val.bin"))

# Save meta information
meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Processing complete. Train: {len(train_ids):,}, Val: {len(val_ids):,}")
