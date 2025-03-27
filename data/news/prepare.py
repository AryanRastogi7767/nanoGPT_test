import os
import pickle

import numpy as np
from datasets import load_dataset

# Load a smaller news dataset (AG News)
dataset = load_dataset("ag_news", split="train")

# Define paths
data_dir = "data/news"
os.makedirs(data_dir, exist_ok=True)
input_file_path = os.path.join(data_dir, "input.txt")


# Process and save data in smaller chunks
def process_and_save(dataset, file_path, chunk_size=5000):
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(0, len(dataset), chunk_size):
            batch_texts = "\n".join(dataset[i : i + chunk_size]["text"])
            f.write(batch_texts + "\n")


process_and_save(dataset, input_file_path)
print(f"Dataset saved at {input_file_path}.")

# Read back the text for processing
with open(input_file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

# Get unique characters
chars = sorted(list(set(text_data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# Function to encode/decode
def encode(s):
    return [stoi.get(c, 0) for c in s]  # Handle unknown chars


def decode(l):
    return "".join([itos.get(i, "?") for i in l])  # Handle unknown indices


# Split data into train and validation sets
n = len(text_data)
train_data = text_data[: int(n * 0.9)]
val_data = text_data[int(n * 0.9) :]

# Encode to integers (Use uint8 if vocab_size <= 256, otherwise use uint16)
dtype = np.uint8 if vocab_size <= 256 else np.uint16
train_ids = np.array(encode(train_data), dtype=dtype)
val_ids = np.array(encode(val_data), dtype=dtype)

# Save binary files
train_ids.tofile(os.path.join(data_dir, "train.bin"))
val_ids.tofile(os.path.join(data_dir, "val.bin"))

# Save meta information
meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Processing complete. Train: {len(train_ids):,}, Val: {len(val_ids):,}")
