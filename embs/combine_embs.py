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
from annoy import AnnoyIndex
from collections import defaultdict
import os

# --- Configuration ---
# Set this to the number of script outputs you have
NUM_CHUNKS = 2  # e.g., if you have files ending in test1, test2, ... test5
EMB_SIZE = 768
METRIC = 'angular' # Must match the metric used to create the original indices

# --- File Naming Convention ---
# Assumes your files are named like this. Change if necessary.
ann_filename_template = "annoy/annoy_tree_qwen_emb_768_test{}.ann"
dict_filename_template = "annoy/annoy_tree_index_to_word_qwen_emb_768_test{}.npy"

def combine_indices(num_chunks: int):
    """
    Loads embeddings from multiple Annoy indices and prepares them for averaging.
    """
    # This dictionary will collect all embeddings for each word.
    # e.g., { "word": [emb_from_chunk1, emb_from_chunk3], ... }
    word_embeddings_collection = defaultdict(list)

    print(f"[*] Starting combination process for {num_chunks} chunks...")

    for i in range(1, num_chunks + 1):
        ann_file = ann_filename_template.format(i)
        dict_file = dict_filename_template.format(i)

        # Check if files for the chunk exist
        if not os.path.exists(ann_file) or not os.path.exists(dict_file):
            print(f"[!] Warning: Data for chunk {i} not found (missing {ann_file} or {dict_file}). Skipping.")
            continue

        print(f"--- Processing Chunk {i} ---")

        # Load the Annoy index
        index = AnnoyIndex(EMB_SIZE, metric=METRIC)
        index.load(ann_file)

        # Load the index-to-word dictionary
        idx_to_word = np.load(dict_file, allow_pickle=True).item()

        # Extract each word and its vector
        for idx, word in idx_to_word.items():
            vector = index.get_item_vector(idx)
            word_embeddings_collection[word].append(vector)

    print("\n[+] Data collection complete. All chunks processed.")
    return word_embeddings_collection

def average_and_build_index(collection: dict):
    """
    Averages the collected embeddings and builds the final Annoy index.
    """
    print("[*] Averaging embeddings and building final index...")

    final_annoy_index = AnnoyIndex(EMB_SIZE, metric=METRIC)
    final_idx_to_word = {}
    item_idx = 0

    # Iterate through each word and its list of embeddings
    for word, embeddings_list in collection.items():
        # Calculate the mean of all embeddings for this word
        if embeddings_list:
            final_embedding = np.mean(embeddings_list, axis=0)

            final_annoy_index.add_item(item_idx, final_embedding)
            final_idx_to_word[item_idx] = word
            item_idx += 1

    print(f"Building final Annoy tree with {final_annoy_index.get_n_items()} unique words...")
    final_annoy_index.build(100, n_jobs=-1) # Build with 100 trees

    # Save the final combined index and word map
    final_annoy_index.save('final_combined.ann')
    np.save('final_combined_idx_to_word.npy', final_idx_to_word)

    print("\n[+] Success! Final combined index saved.")
    print("Files: 'final_combined.ann' and 'final_combined_idx_to_word.npy'")

if __name__ == "__main__":
    # 1. Collect all embeddings from all files
    embeddings_collection = combine_indices(NUM_CHUNKS)

    # 2. Average them and build the new index
    if embeddings_collection:
        average_and_build_index(embeddings_collection)
    else:
        print("[!] No data was found to process. Please check your file paths and NUM_CHUNKS.")




# In[ ]:




