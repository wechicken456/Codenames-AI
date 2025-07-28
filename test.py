#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[ ]:


word = "car"
r = requests.get(f"http://api.conceptnet.io/c/en/{word}").json()


# In[7]:


r.keys()


# In[8]:


len(r["edges"])


# In[9]:


r["edges"]


# In[14]:


noun_rels = [
    "RelatedTo", 
    "CapableOf", 
    "IsA", 
    "UsedFor", 
    "AtLocation", 
    "HasPrerequisite",
    "HasProperty", 
    "ReceivesAction",
    "CreatedBy",
    "Causes",
    "HasA"
    "MadeOf",
]

noun_rev_rels = [
    "AtLocation",
    "IsA",
    "PartOf"
]

verb_rels = [
    "MannerOf", 
    "HasSubevent",
    "MotivatedByGoal",
    "IsA",
    "HasFirstSubevent",
    "HasLastSubevent",
]
verb_rev_rels = [
    "CapableOf",
    "MannerOf",
    "CausesDesire",
    "CreatedBy",
    "UsedFor",
    "ReceivesAction"
]


# In[2]:


import spacy

nlp = spacy.load("en_core_web_trf")
doc = nlp("San Francisco considers banning sidewalk delivery robots")

# document level
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print(ents)

# token level
ent_san = [doc[0].text, doc[0].ent_iob_, doc[0].ent_type_]
ent_francisco = [doc[1].text, doc[1].ent_iob_, doc[1].ent_type_]
print(ent_san)  # ['San', 'B', 'GPE']
print(ent_francisco)  # ['Francisco', 'I', 'GPE']


# In[ ]:





# In[ ]:




