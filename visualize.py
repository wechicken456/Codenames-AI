#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas
import csv
import re
import string
import os


# In[3]:


df = pandas.read_csv("./SWOW-EN.R100.csv")


# In[88]:


df = df[~df["R1"].isna()]
r1 = df["R1"]

l = r1.apply(lambda x: len(x.split(" ")))
t = l.apply(lambda x : x - 1)
t = t[t > 0]
print(t)


# In[91]:


df.iloc[t.index]


# In[22]:


swow_rel_forward = "forwardassociated"
swow_rel_bidirectional = "bidirectionalassociated"
swow_rel_backward = "backwardassociated"


relation_groups = [
  swow_rel_forward,
  swow_rel_bidirectional 
]

merged_relations = [
   swow_rel_forward,
   swow_rel_bidirectional,
]

relation_groups_1rel= [
  swow_rel_forward,
]

merged_relations_1rel = [
   swow_rel_forward,
]

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

class SWOW(object):
    def __init__(self, swow_file, output_csv_path="./data/swow/swow_associations.csv",  output_vocab_path="./data/swow/swow_vocab.csv", kg_name='swow', word_pair_freq=1):
        self.kg_name = kg_name 
        self.swow_data = self.load_swow_en(swow_file)
        self.swow_cue_responses, self.concepts = self.forward_associations(self.swow_data, word_pair_freq, unify_nodes=True)
        self.swow_cue_responses_relation = self.add_relations(self.swow_cue_responses)
        if output_csv_path is not None:
            self.write_cues(self.swow_cue_responses.keys(), output_path="./data/swow/swow_cues.csv")
            self.write_forward_associations_relation(self.swow_cue_responses_relation, output_csv_path, output_vocab_path)

    def load_swow_en(self, input_file):
        cues, R1, R2, R3 = list(),list(),list(),list()
        reader =csv.DictReader(open(input_file))
        for row in reader:
            cues.append(row['cue'].lower())
            R1.append(row['R1'].lower())
            R2.append( row['R2'].lower())
            R3.append( row['R3'].lower())

        swow_data = list(zip(cues, R1, R2, R3))
        print("Loaded %d lines from %s"%(len(cues),input_file))
        return swow_data

    def unify_sw_nodes(self, node):
        '''unify entity format with ConceptNet5.6, in which entity is concatenated by words with _'''
        '''keep words concatenated by -, like 'self-esteem', 'self-important' '''
        node_list_raw = re.split(' ', node)

        blacklist = ['a'] # a club,
        if len(node_list_raw)>1 and node_list_raw[0] in blacklist:
            node_list_raw.remove(node_list_raw[0])

        if node_list_raw[0].startswith("-"): #-free (gluten -free)
            node_list_raw[0] = node_list_raw[0][1:]

        if node_list_raw[0].startswith("_"): #_position
            node_list_raw[0] = node_list_raw[0][1:]

         #cases: beard_-_eyebrows_-_mustache,  bearskin___________disrobe_________reveal, bear__wine
        node_list =  []
        for node in node_list_raw:
            node = node.replace("_-_", "")
            node = node.replace("___________", "")
            node = node.replace("__", "")
            node = node.replace("_", "")
            node = node.replace("__","")
            node = node.replace("__","")
            #node = node.replace("-", "_") #real text contains -, eg, self-important
            if node: # if not empty string, "- Johnson wife of lyndon"
                node_list.append(node)

        node_len = len(node_list)
        if node_len >0:
            node_phrase = "_".join(node_list)
            #if not en_dict.check(node_phrase):
            #   print(node_phrase)
            return node_phrase, node_len
        else: #filter empty node
            #print("empty node: {}".format(node_list_raw))
            return None, None

    def forward_associations(self, swow_data, word_pair_freq, unify_nodes=False):
        cue_responses={}
        concepts=set()
        phrase_seen = dict()
        for i, (cue, r1, r2, r3) in enumerate(swow_data):

            cue = cue.lower()
            if unify_nodes:
                phrase_ori = cue
                cue, phrase_len = self.unify_sw_nodes(cue)
                if phrase_len is None:
                    continue
                if phrase_len >1:
                    if cue not in phrase_seen:
                        phrase_seen[cue]=[phrase_ori]
                    else:
                        phrase_seen[cue].extend([phrase_ori])

            for r in [r1, r2, r3]:
                #if cue not in cue_responses.keys() and r!="NA" or "na":
                r = r.lower()

                if r=='na' or r == "nan": continue
                if cue == r: continue #about 1000, e.g., read, aimless, sheen, elbows

                if unify_nodes:
                    phrase_ori = r
                    r, phrase_len = self.unify_sw_nodes(r)

                    if phrase_len is None:
                        continue

                    if phrase_len >1:
                        if r not in phrase_seen:
                            phrase_seen[r]=[phrase_ori]
                        else:
                            phrase_seen[r].extend([phrase_ori])

                if not cue.replace("_", "").replace("-", "").replace(" ","").replace("''","").isalpha():
                    #print("cue: {}".format(cue))
                    continue
                if not r.replace("_", "").replace("-", "").replace(" ","").replace("''","").isalpha():
                    #print("response: {}".format(r))
                    continue

                if cue in string.punctuation or r in string.punctuation:
                    print(f"dirty data: {cur}, {r}")
                    continue

                if cue not in cue_responses.keys() :
                    cue_responses[cue]={r:1}
                    concepts.add(cue)
                    concepts.add(r)
                else:
                    cue_responses = self.add_elements(cue_responses, cue, r)
                    concepts.add(r)


        num_swow_triplets = sum([len(x) for x in cue_responses.values()])
        print("Number of original triplets in SWOW is {}".format(num_swow_triplets))
        if word_pair_freq >1:
            cue_responses = self.filter_frequency(cue_responses, word_pair_freq)
            cut_down_num = num_swow_triplets - sum([len(x) for x in cue_responses.values()])
            print("Cutting down {} triplets whose wordpair_frequency<{}".format(cut_down_num, word_pair_freq))

            num_swow_triplets = sum([len(x) for x in cue_responses.values()])
            print("Number of original triplets in SWOW is {} (after cutting down)".format(num_swow_triplets))

        return cue_responses, concepts

    def add_relations(self, cue_responses):
        cue_responses_relation= list() # bugfix: use list() instead of set(), guaranteeing vocab order to be the same for everytime 
        count_bi = 0
        count_fw = 0
        for cue, vs in cue_responses.items():
            for response, freq in vs.items():
                rel_forward = swow_rel_forward.lower()
                cue_responses_relation.append((rel_forward, cue, response, freq))
                count_fw +=1

                if self.kg_name == 'swow':
                    if response in cue_responses and cue in cue_responses[response]:
                        rel_bidirection = swow_rel_bidirectional.lower()
                        cue_responses_relation.append((rel_bidirection, cue, response, freq))
                        count_bi+=1
        print("Add {} forward association triples".format(count_fw))
        print("Add {} bi-directional association triples".format(count_bi))
        return cue_responses_relation

    def write_cues(self, cues, output_path):
        check_path(output_path)
        with open(output_path, 'w') as fout:
            for cue in cues:
              fout.write(cue+'\n')
        print("write {} {} cues".format(output_path, len(cues)))

    def write_forward_associations_relation(self, cue_responses_relation,
                    output_csv_path, output_vocab_path):
        '''
        input: (rel, heat, tail, freq)
        '''
        cpnet_vocab = []
        # cpnet_vocab.append(PAD_TOKEN)

        concepts_seen = set()
        check_path(output_csv_path)
        fout = open(output_csv_path, "w", encoding="utf8")
        # cue_responses_relation = list(cue_responses_relation)
        cnt=0
        for (rel, head, tail, freq) in cue_responses_relation:
            fout.write('\t'.join([rel, head, tail, str(freq)]) + '\n')
            cnt+=1
            for w in [head, tail]:
                if w not in concepts_seen:
                    concepts_seen.add(w)
                    cpnet_vocab.append(w)

        check_path(output_vocab_path)
        with open(output_vocab_path, 'w') as fout:
            for word in cpnet_vocab:
                fout.write(word + '\n')

        print('extracted {} triples to {}'.format(cnt, output_csv_path))
        print('extracted {} concpet vocabulary to {}'.format(len(cpnet_vocab), output_vocab_path))
        print()

        return cpnet_vocab


    def add_elements_dict2d(self,outter, outter_key, inner_key,value):
        if outter_key not in outter.keys():
            outter.update({outter_key:{inner_key:value}})
        else:
            outter[outter_key].update({inner_key:value})
        return outter

    def add_elements(self,outter, outter_key, inner_key):
        if inner_key not in outter[outter_key].keys():
            outter[outter_key].update({inner_key:1})
        else:
            outter[outter_key][inner_key]+=1
        return outter

    def filter_frequency(self,cue_responses, word_pair_freq=2):
        new_cue_responses={}

        for i, (cue,responses) in enumerate(tqdm(cue_responses.items())):
            for response,frequency in responses.items():
                if response == 'NA' or response=='na': continue

                if frequency >= word_pair_freq:
                    self.add_elements_dict2d(outter=new_cue_responses,
                                        outter_key=cue,
                                        inner_key=response,
                                        value=frequency)
        return new_cue_responses


# In[23]:


SWOW("./SWOW-EN.R100.csv")


# In[ ]:





# In[ ]:





# In[140]:


df = pandas.read_csv("./data/swow/swow_associations.csv", sep = "\t")


# In[158]:


df = df.rename(columns = {"forwardassociated": "associtation_type", "although": "word1", "nevertheless": "word2", "3": "count"})
df[(df["word1"] == "star") &  (df["word2"] == "moon")]


# In[96]:


df = pandas.read_csv("./swow.bidirectionalassociated.csv", sep = "\t")


# In[97]:


df


# In[102]:


with open("./wikipedia_freq.txt", "r") as f:
    lines = f.readlines()
    lines = [line.split(" ") for line in lines]
    wiki_freq = dict(lines)
    wiki_freq = {k: int(v) for k, v in wiki_freq.items()}




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




