#!/usr/bin/env python
# coding: utf-8

# In[2]:


from annoy import AnnoyIndex
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")


# In[ ]:





# In[ ]:


def split_articles_into_paragraphs(batch):
    paragraphs = []
    for article in batch["text"]:
        paragraphs += article.split("\n\n")
    return {"paragraph": paragraphs}


def clean_dataset():
    start = time.time()
    print("[*] Loading dataset...")
    dataset = load_dataset("omarkamali/wikipedia-monthly", "latest.en", num_proc=16, split="train")
    print(dataset)
    print(f"[+] Finished loading dataset. Time taken: {time.time() - start}")

    print("[*] Splitting into paragraphs...")
    cleaned_dataset = dataset.map(split_articles_into_paragraphs, batched=True, batch_size=768, num_proc = 16, remove_columns=dataset.column_names)
    print("[+] Done splitting into paragraphs")
    print("[*] Saving cleaned dataset to disk...")
    output_path = "./cleaned_paragraphs"
    cleaned_dataset.save_to_disk(output_path)
    print(f"Total time taken: {time.time() - start:.2f}s")


# In[ ]:


clean_dataset()


# In[ ]:





# In[ ]:





# In[ ]:




