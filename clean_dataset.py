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
from datasets import load_dataset, concatenate_datasets, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")


# In[2]:


nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# In[ ]:


def split_articles_into_sentences(batch):
    docs = nlp.pipe(batch["text"], n_process=12, batch_size=64)
    sentences = [sent.text.strip() for doc in docs for sent in doc.sents]
    return {"sentence": sentences}


def clean_dataset():
    start = time.time()
    print("[-] Loading dataset...")
    dataset = load_dataset("omarkamali/wikipedia-monthly", "latest.en", num_proc=16, split="train")
    print(dataset)
    print(f"[+] Finished loading dataset. Time taken: {time.time() - start}")

    print("[-] Splitting into sentences...")
    sentencized_dataset = dataset.map(split_articles_into_sentences, batched=True, batch_size=768, remove_columns=dataset.column_names)
    print("[+] Done splitting into sentences")
    print("[-] Saving cleaned dataset to disk...")
    output_path = "./cleaned_sentences"
    sentencized_dataset.save_to_disk(output_path)
    print(f"Total time taken: {time.time() - start:.2f}s")


# In[ ]:


clean_dataset()

