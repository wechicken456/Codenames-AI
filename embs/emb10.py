#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !python -V
# !pip show torch
# !nvcc --version


# In[ ]:


# !pip install annoy
# !pip install paragraph-transformers
# !pip install transformers
# !pip install datasets
# !pip install spacy


# In[ ]:


# !python -m spacy download en_core_web_trf


# In[ ]:


# # !wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# # !pip install flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# !pip uninstall -y flash-attn


# In[20]:


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[1]:


dataset = load_from_disk("./cleaned_paragraphs")
print("Dataset: ", dataset)
print(f"Len of dataset: {len(dataset)}")


# In[3]:


nlp = spacy.load("en_core_web_trf", disable=["parser"])


# In[ ]:


print("[-] Loading embedding model...")
model_name = "Qwen/Qwen3-Embedding-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
nlp.add_pipe('sentencizer')
print("[+] Finished loading embedding model...")
print(f"[+] Model using device: {model.device}, dtype: {model.dtype}")


# In[19]:


stopwords = set([
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
])

# (averaged_embedding, count_of_word) at index i in embedding_vector_averages is the average qwen embedding of a word.
# The word to index map is stored in word_to_idx_dict
embedding_vector_averages = []
idx = 0
word_to_idx_dict = dict()
bank_idx = 1
emb_size = 768
debug = False

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def add_emb_batch(paragraphs):
    '''
    paragraphs = batch of single paragraphs
    '''
    global idx
    global word_to_idx_dict
    global embedding_vector_averages
    global bank_idx
    global emb_size

    try:
        inputs = tokenizer(paragraphs, 
                           return_tensors="pt",
                           padding = True, 
                           truncation = True, 
                           max_length = 16834, 
                           return_offsets_mapping=True)

        offset_mappings = inputs.pop("offset_mapping")
        inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1) # normalize over the last dim (emb dim)
        embeddings = embeddings.cpu()
        # shape of embeddings: (batch size, paragraph length, hidden size)
        embeddings = embeddings[:, :, :emb_size]

        spacy_docs = nlp.pipe(paragraphs)

        for i, paragraph_doc in enumerate(spacy_docs):
            paragraph_embeddings = embeddings[i]
            paragraph_offsets = offset_mappings[i]

            # OPTIMIZATION: Index 'j' tracks our position in paragraph_offsets.
            j = 0

            # print(f"---------------- IDX {i} -------------------")
            # print(f"offset_mappings{i}: {paragraph_offsets}")
            start = time.time()
            #print(f"Processing paragraph: {paragraph_doc.text}")

            for token in paragraph_doc:

                word = token.text.lower()

                # only create embeddings for words that only contain alphabetical characters
                if not word.isalpha():
                    continue

                start_char = token.idx
                end_char = token.idx + len(token.text)


                if token.is_punct or token.is_space or token.text.lower() in stopwords:
                    continue

                #print(f"\rProcessing word: '{word}' (chars {start_char}-{end_char})")


                # -------------- Find all transformer tokens/sub-tokens that encompass (cover) this spaCy token's span entirely ------------------


                # OPTIMIZATION: Advance 'j' to the first possible sub-token for the current word.
                # This avoids re-scanning from the beginning of the list for every word.
                while j < len(paragraph_offsets) and paragraph_offsets[j][1] <= start_char:
                    j += 1

                word_tokens_indices = []
                temp_j = j


                # Find all sub-tokens that overlap with the current spaCy token
                while temp_j < len(paragraph_offsets):
                    offset_start, offset_end = paragraph_offsets[temp_j]

                    # If the sub-token starts after the word ends, we can stop searching.
                    if offset_start >= end_char:
                        break

                    # Check for overlap and that it's not a [0,0] padding token.
                    if offset_end > start_char and offset_end > offset_start:
                        word_tokens_indices.append(temp_j)

                    temp_j += 1

                if not word_tokens_indices:
                    continue

                # get embeddings for tokens that make up this word.
                embedding = paragraph_embeddings[word_tokens_indices]
                embedding = F.normalize(embedding, p = 2, dim = -1)
                embedding = embedding.mean(dim=0)  # shape: (emb_size,)

                # First time we've seen this word
                if word not in word_to_idx_dict:
                    word_to_idx_dict[word] = idx
                    embedding_vector_averages.append((embedding, 1))
                    idx += 1
                # Otherwise, average this embedding with the running average
                else:

                    curr_word_idx = word_to_idx_dict[word]
                    curr_average_vector = embedding_vector_averages[curr_word_idx][0]
                    curr_count = embedding_vector_averages[curr_word_idx][1]

                    averaged_embedding = np.average( np.array([ curr_average_vector, embedding ]), axis=0, weights = [curr_count, 1] )
                    embedding_vector_averages[curr_word_idx] = (averaged_embedding, curr_count + 1)

                #print(f"\r'{word}' ----->:")
                #print("\rembedding (first 5 dims): ", embedding[:5])

            #print(f"Time taken to process paragraph: {time.time() - start:2f}")


    except Exception as e:
        #print("Could not get qwen embedding for paragraphs ", paragraphs)
        print(e)

def process_batch(batch):
    add_emb_batch(batch["paragraph"])

def create_embeddings(script_id : int):
    #ctx = mx.gpu(0)

    if debug:
        print("[*] Splitting into paragraphs...")
        paragraphs = dataset["paragraph"][:512] 
        subset = Dataset.from_dict({"paragraph" : paragraphs})
        print("[+] Done splitting into paragraphs")
        subset.map(process_batch, batched=True, batch_size=32)    
    else:
        shuffled_dataset = dataset.shuffle(seed=42)["paragraph"][15_750_000:17_500_000]
        truncated_dataset = Dataset.from_dict({"paragraph": shuffled_dataset})
        truncated_dataset.map(process_batch, batched=True, batch_size=64)

    print("Number of embeddings", len(embedding_vector_averages))
    print("Number of words in word_to_idx_dict", len(word_to_idx_dict.keys()))

   # Convert list of tuples to a more efficient numpy array for embeddings and a list for counts
    final_embeddings = np.array([item[0] for item in embedding_vector_averages], dtype=np.float16)
    final_counts = np.array([item[1] for item in embedding_vector_averages], dtype=np.int32)

    idx_to_word_dict = {v: k for k, v in word_to_idx_dict.items()}

    output_prefix = f"chunk_{script_id}"
    np.save(f"{output_prefix}_embeddings.npy", final_embeddings)
    np.save(f"{output_prefix}_counts.npy", final_counts)
    with open(f"{output_prefix}_word_to_idx.pkl", "wb") as f:
        pickle.dump(word_to_idx_dict, f)

if __name__=='__main__':
    start = time.time()
    create_embeddings(10)
    print(f"Time taken to create embeddings: {time.time() - start}")


# In[ ]:




