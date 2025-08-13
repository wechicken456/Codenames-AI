#!/usr/bin/env python
# coding: utf-8

# In[7]:


from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import time
from sentence_transformers import SentenceTransformer
import json
import asyncio
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Download necessary data for WordNetLemmatizer if we haven't already
try:
    WordNetLemmatizer().lemmatize("test") # Just a test to trigger lookup error if not downloaded
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual Wordnet, often needed for full WordNet functionality


# In[ ]:





# In[9]:


class LLM:
    def __init__(self, openAI_api_key):
        self.client = AsyncOpenAI(api_key=openAI_api_key)


    def _get_word_classes(self, word : str) -> set:
        pos_set = set()
        for synset in wn.synsets(word):
            if synset.name().split('.')[0] != word:
                continue  # Only consider exact matches
            if synset.pos() == 'n':
                pos_set.add('noun')
            elif synset.pos() == 'v':
                pos_set.add('verb')
            elif synset.pos() == 's':
                pos_set.add('adj')

        if len(pos_set) == 0: pos_set.add('noun') # default to noun if unidentified 
        return pos_set

    def LLM_response_to_JSON(self, resp : str) -> dict:
        try:
            resp = json.loads(resp)
            return resp
        except:
            return {}

    def extract_clues_from_LLM_response(self, resp: dict) -> set:
        try:
            relations = resp["example_relations"]
            clues = set()
            for rel in relations:
                _clues = [rel[-3]] + [rel[-2]] + [rel[-1]]
                clues.add(_clues)
            return clues
        except:
            return set()

    def extract_sentences_from_LLM_response(self, resp: str) -> list[str]:
        try:
            relations = resp["example_relations"]
            sents = []
            for rel in relations:
                _sents = [rel[1]] + [rel[2]] + [rel[3]]
                sents += _sents
            return sents
        except:
            return []



    async def get_clues_for_word(self, target_word : str, assassin_word : str):
        """
        returns (target_word, list of strings of clues for this word)
        """
        pos_list = list(self._get_word_classes(target_word))
        pos_str = "[" + ", ".join(pos_list) + "]"
        prompt = f"""
        **Objective:**
        You are a Linguistic Knowledge AI. Your task is to generate a list of simple, definitional example sentences for a given English word. Each example must connect strongly to commonsense English knowledge such that an average college graduate (no specific major) can understand it - no extremely niche references. 
        You will consider all of its specified parts of speech and generate sentences based on a comprehensive list of semantic relations.
        Be as quick as you can.

        **Instructions:**
        1.  Analyze the target word: **{target_word}**
        2.  Consider its meaning for all following roles: **{pos_str}**.
        3.  For each part of speech, generate three clear example sentence for each of its applicable relations and reverse relations. Each example must be distinct, use commonsense knowledge in the English language, and has a strong connection to the target word **{target_word}**.
        4.  Strictly follow the sentence structure provided for each relation in the "Relation Definitions" section below.
        5.  If a relation is not applicable for the target word, simply skip it.
        7.  Your final response must be a single JSON object with one key, "example_relations". Do not include any other text or explanation. Include the nodes A and B in each relation.
        8.  Your examples must in no way relate to the word '{assassin_word}'.
        9.  ONLY use English words.

        **Relation Definitions (from ConceptNet):**
        """

        pos_rels = ""
        if "noun" in pos_list or "NOUN" in pos_list:
            pos_rels += f"""
            ### NOUN Relations (A -> B)
            * **IsA**: A is a subtype or a specific instance of B; every A is a B. This is the hyponym relation. Structure: "The noun {target_word} is a type of [concept]."
            * **UsedFor**: A is used for B; the purpose of A is B. Structure: "The noun {target_word} is used for [purpose or action]."
            * **CapableOf**: Something A can typically do is B. Structure: "The noun {target_word} is capable of [action]."
            * **AtLocation**: A can typically be found at B. Structure: "The noun {target_word} is found at/in/on [location]"
            * **HasProperty**: A has B as a property. Structure: "The noun {target_word} has the property of being [adjective]."
            * **HasA**: A is a whole which has B as a very well-known/typical part. Structure: "The noun {target_word} has a [part]."
            * **MadeOf**: A is made of the substance B. Structure: "The noun {target_word} is made of [substance]."
            * **ReceivesAction**: B is an action that can be done to A. Structure: "The noun {target_word} can be [action, e.g., 'driven' or 'eaten']."
            * **CreatedBy**: B is a process or agent that creates A. Structure: The noun {target_word} is created by [process or person]."
            * **Causes**: A and B are events, and it is typical for A to cause B. Structure: "The noun {target_word} causes [event or state]."
            * **HasPrerequisite**: In order for A to happen, B needs to happen. Structure: "The noun {target_word} has a prerequisite of [concept]."
            * **AssociatedNames**: B is a well-known name of an entity (person, character, location, book, etc) that is very commonly (according to an average person) associated with A. Structure "The name [name] ...".

            ### NOUN Reverse Relations (B -> A)
            * **IsA (Reverse)**: B is a supertype of A; every A is a B. Structure: "[An example] is a type of {target_word}."
            * **AtLocation (Reverse)**: B is a location where A can be found. Structure: "[An object] is found in {target_word}."
            * **PartOf**: B is a whole that A is a part of; A is a component of B. This is the meronym relation. Structure: "[An object] is a part of {target_word}."
            """
        if "verb" in pos_list or "VERB" in pos_list:
            pos_rels += f"""
            ### VERB Relations (A -> B)
            * **IsA**: A is a subtype of B; to do A is one way to do B. Structure: "To {target_word} is a way to [more general action]."
            * **MannerOf**: A is a specific manner of doing B. Structure: "To {target_word} is a manner of [more general action]."
            * **HasSubevent**: A is an event that has B as a subevent. Structure: "A subevent of the action {target_word} is [component action]."
            * **HasFirstSubevent**: A is an event whose first subevent is B. Structure: "The first subevent of the action {target_word} is [component action]."
            * **HasLastSubevent**: A is an event whose last subevent is B. Structure: "The last subevent of the action {target_word} is [component action]."
            * **MotivatedByGoal**: Someone does A because they want to achieve goal B. Structure: "The action {target_word} is motivated by the goal of [goal]."

            ### VERB Reverse Relations (B -> A)
            * **MannerOf (Reverse)**: B is a more general action that can be done in the specific manner A. Structure: "[A specific action] is a manner of {target_word}."
            * **CapableOf (Reverse)**: B is an action that can be performed by A. Structure: "[An object] is capable of the action {target_word}."
            * **UsedFor (Reverse)**: B is an action that is the purpose of A. Structure: "[An object] is used for the action {target_word}."
            * **ReceivesAction (Reverse)**: B is an action that can be done to A. Structure: "[An object] can receive the action {target_word}."
            * **CreatedBy (Reverse)**: B is an action that creates A. Structure: "[A result] is created by the action {target_word}."
            * **CausesDesire (Reverse)**: B is an action that one desires to do because of A. Structure: "[A feeling or concept] makes a person want to {target_word}."
            * **HasFirstSubevent (Reverse)**: B is a larger event whose first subevent is A. Structure: "{target_word} is the first subevent of [a larger action]."
            * **AssociatedNames**: B is a well-known name of an entity (person, character, location, book, etc) that is very commonly known to do A. Structure "The name [name] ...".
            """
        if "adj" in pos_list or "ADJ" in pos_list:
            pos_rels += f"""
            ### ADJECTIVE Relations (A -> B)
            * **SimilarTo**: A is an attribute that is similar to attribute B. Structure: "To be {target_word} is similar to being [adjective]."
            * **RelatedTo**: A is an attribute that is related to concept B. Structure: "Being {target_word} is related to [concept]."
            * **HasProperty**: The state of being A has the property B. Structure: "A property of being {target_word} is [concept]."
            """
        prompt += pos_rels
        prompt += f"""
        ---
        **TASK**

        **Target Word:** {target_word}
        **Parts of Speech to Consider:** {pos_str}
        **Required JSON Output Format:**
        {{
          "example_relations": [
            ["{{name of the relation}}", "sentence 1", "sentence 2", sentence 3", "{{The single most important word in node B in the 1st example}}", "{{The single most important word in node B in the 2nd example}}", "{{The single most important word in node B in the 3rd example}}"],
            ["{{name of the relation}}", "sentence 2", "sentence 2", sentence 3", "{{The single most important word in node B in the 1st example}}", "{{The single most important word in node B in the 2nd example}}", "{{The single most important word in node B in the 3rd example}}"], 
            ["{{name of the relation}}", "sentence 3", "sentence 2", sentence 3", "{{The single most important word in node B in the 1st example}}", "{{The single most important word in node B in the 2nd example}}", "{{The single most important word in node B in the 3rd example}}"] 
          ]
        }}
        """

        history = [{"role": "user", "content": prompt}]
        response = await self.client.chat.completions.create(
            messages=history,
            model="gpt-4.1",
            response_format={ "type": "json_object" }
        )
        return target_word, self.extract_clues_from_LLM_response(response.choices[0].message.content)

    async def get_clues_for_words(self, wordlist : list[str], assassin_word : str) -> list:
        """
        Create multiple parallel OpenAI API requetsts to get clues for each word in `wordlist`.
        Each target word will produce a set of clues for that particular word.
        Return a list of set of clues.
        """
        tasks = []
        for word in wordlist:
            tasks.append(self.get_clues_for_word(word, assassin_word))

        res = await asyncio.gather(*tasks)
        return res



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




