#!/usr/bin/env python
# coding: utf-8

# In[1]:


from annoy import AnnoyIndex
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import spacy
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
import pickle
from annoy import AnnoyIndex
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


import numpy as np
import pickle
from annoy import AnnoyIndex
import os

EMB_SIZE = 768
NUM_CHUNKS = 16
METRIC = 'angular'

def combine_embeddings(num_chunks: int):
    """
    Combines embedding data from multiple chunks into a single,
    globally-averaged set of embeddings.
    """
    # This dictionary will store the final combined data:
    # { "word": (running_average_embedding, total_count) }
    combined_data = {}

    print(f"[*] Starting combination process for {num_chunks} chunks...")

    for i in range(1, num_chunks + 1):
        prefix = f"chunk_{i}"

        if not os.path.exists(f"{prefix}_embeddings.npy"):
            print(f"[!] Warning: Data for chunk {i} not found. Skipping.")
            continue

        print(f"--- Processing Chunk {i} ---")

        chunk_embeddings = np.load(f"{prefix}_embeddings.npy")
        chunk_counts = np.load(f"{prefix}_counts.npy")
        with open(f"{prefix}_word_to_idx.pkl", "rb") as f:
            chunk_word_to_idx = pickle.load(f)

        for word, idx in chunk_word_to_idx.items():
            chunk_emb = chunk_embeddings[idx]
            chunk_count = chunk_counts[idx]

            # Check if the incoming embedding from the chunk is valid.
            # If it's not, we skip it to prevent corrupting the final average.
            if np.isnan(chunk_emb).any():
                print(f"[!] Warning: NaN found for word '{word}'. Skipping...")
                continue 

            # If we've seen this word before, perform a weighted average
            if word in combined_data:
                total_emb, total_count = combined_data[word]

                # New average = ( (old_avg * old_count) + (new_avg * new_count) ) / (old_count + new_count)
                new_total_count = total_count + chunk_count

                new_avg_emb = np.average(
                    [total_emb, chunk_emb], 
                    axis=0, 
                    weights=[total_count, chunk_count]
                )

                combined_data[word] = (new_avg_emb, new_total_count)

            else:
                combined_data[word] = (chunk_emb, chunk_count)

    print("\n[+] Combination complete. All chunks merged.")
    return combined_data

def build_final_annoy_index(combined_data: dict):
    """
    Builds and saves a final Annoy index from the combined data.
    """
    print("[*] Building final Annoy index...")

    t = AnnoyIndex(EMB_SIZE, metric=METRIC)
    final_idx_to_word = {}

    for i, (word, (embedding, count)) in enumerate(combined_data.items()):
        if len(embedding) == EMB_SIZE:
            t.add_item(i, embedding)
            final_idx_to_word[i] = word

    print(f"Building Annoy tree with {t.get_n_items()} items...")
    t.build(200, n_jobs=-1) # Build with 100 trees

    # Save the final index and the word mapping
    t.save('final_combined.ann')
    np.save('final_combined_idx_to_word.npy', final_idx_to_word)

    print("[+] Annoy index and word map created and saved successfully!")
    print("Files: 'final_combined.ann' and 'final_combined_idx_to_word.npy'")

if __name__ == "__main__":
    start = time.time()
    final_data = combine_embeddings(NUM_CHUNKS)
    if final_data:
        build_final_annoy_index(final_data)

    print(f"Time taken: {time.time() - start}")


# In[ ]:




