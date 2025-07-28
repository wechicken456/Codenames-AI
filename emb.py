#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !python -V
# !pip show torch
# !nvcc --version


# In[ ]:


# !pip install annoy
# !pip install sentence-transformers
# !pip install transformers
# !pip install datasets
# !pip install spacy


# In[ ]:


# !python -m spacy download en_core_web_trf


# In[ ]:


# # !wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# # !pip install flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# !pip uninstall -y flash-attn


# In[14]:


from annoy import AnnoyIndex
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import spacy
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


dataset = load_from_disk("./cleaned_sentences")


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


def add_emb_batch(sentences):
    '''
    sentences = batch of single sentences
    '''
    global idx
    global word_to_idx_dict
    global embedding_vector_averages
    global bank_idx
    global emb_size

    try:
        inputs = tokenizer(sentences, 
                           return_tensors="pt",
                           padding = True, 
                           truncation = True, 
                           max_length = 128, 
                           return_offsets_mapping=True)

        offset_mappings = inputs.pop("offset_mapping")
        inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1) # normalize over the last dim (emb dim)
        embeddings = embeddings.cpu()
        # shape of embeddings: (batch size, sentence length, hidden size)
        embeddings = embeddings[:, :, :emb_size]

        spacy_docs = nlp.pipe(sentences)

        for i, sentence_doc in enumerate(spacy_docs):
            sentence_embeddings = embeddings[i]
            sentence_offsets = offset_mappings[i]

            # OPTIMIZATION: Index 'j' tracks our position in sentence_offsets.
            j = 0

            # print(f"---------------- IDX {i} -------------------")
            # print(f"offset_mappings{i}: {sentence_offsets}")
            start = time.time()
            #print(f"Processing sentence: {sentence_doc.text}")

            for token in sentence_doc:

                word = token.text.lower()
                start_char = token.idx
                end_char = token.idx + len(token.text)


                if token.is_punct or token.is_space or token.text.lower() in stopwords:
                    continue

                #print(f"\rProcessing word: '{word}' (chars {start_char}-{end_char})")


                # -------------- Find all transformer tokens/sub-tokens that encompass (cover) this spaCy token's span entirely ------------------


                # OPTIMIZATION: Advance 'j' to the first possible sub-token for the current word.
                # This avoids re-scanning from the beginning of the list for every word.
                while j < len(sentence_offsets) and sentence_offsets[j][1] <= start_char:
                    j += 1

                word_tokens_indices = []
                temp_j = j


                # Find all sub-tokens that overlap with the current spaCy token
                while temp_j < len(sentence_offsets):
                    offset_start, offset_end = sentence_offsets[temp_j]

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
                embedding = sentence_embeddings[word_tokens_indices]
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

            #print(f"Time taken to process sentence: {time.time() - start:2f}")


    except Exception as e:
        #print("Could not get qwen embedding for sentences ", sentences)
        print(e)

def process_batch(batch):
    add_emb_batch(batch["sentence"])

def create_index():
    #ctx = mx.gpu(0)

    if debug:
        print("[-] Splitting into sentences...")
        sentences = dataset["sentence"][:1024] 
        subset = Dataset.from_dict({"sentence" : sentences})
        print("[+] Done splitting into sentences")
        subset.map(process_batch, batched=True, batch_size=128)    
    else:
        shuffled_dataset = dataset.shuffle(seed=42)["sentence"][:3_000_000]
        truncated_dataset = Dataset.from_dict({"sentence": shuffled_dataset})
        truncated_dataset.map(process_batch, batched=True, batch_size=64)

    print("Number of embeddings", len(embedding_vector_averages))
    print("Number of words in word_to_idx_dict", len(word_to_idx_dict.keys()))

    tree_idx = 0
    mod = 50000
    t = AnnoyIndex(emb_size, metric='angular')

    for x in range(len(embedding_vector_averages)):
        if (tree_idx % mod == 0): print("ADDED ", tree_idx, " EMBEDDINGS TO ANNOY TREE")

        embedding = embedding_vector_averages[x][0]

        if len(embedding) == 0:
            continue

        t.add_item(x, embedding)

        tree_idx += 1

    t.build(100, n_jobs=-1)

    idx_to_word_dict = {v: k for k, v in word_to_idx_dict.items()}

    t.save('annoy_tree_qwen_emb_768_test.ann')
    np.save('annoy_tree_index_to_word_qwen_emb_768_test.npy', idx_to_word_dict)


def test():
    # test the index
    index = AnnoyIndex(emb_size, metric='angular')
    index.load('annoy_tree_qwen_emb_768_test.ann')
    annoy_tree_idx_to_word = np.load('annoy_tree_index_to_word_qwen_emb_768_test.npy', allow_pickle=True).item()
    word_to_idx_dict = {v: k for k, v in annoy_tree_idx_to_word.items()}

    start = time.time()
    print(f"TESTING neighbors for the word 'ice':")
    inp_idx = word_to_idx_dict["drive"]
    related_list = index.get_nns_by_item(inp_idx, 10)
    for i in related_list:
        print(annoy_tree_idx_to_word[i])
    print()
    print(f"Time taken: {time.time() - start:2f}")

if __name__=='__main__':
    create_index()
    test()


# In[ ]:




