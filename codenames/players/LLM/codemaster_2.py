#!/usr/bin/env python
# coding: utf-8

# In[2]:


from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import time
from sentence_transformers import SentenceTransformer
import json
import asyncio
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# Download necessary data for WordNetLemmatizer if we haven't already
try:
    WordNetLemmatizer().lemmatize("test") # Just a test to trigger lookup error if not downloaded
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual Wordnet, often needed for full WordNet functionality


# In[ ]:





# In[55]:


class LLM2:
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
            clue_list = resp["clues"]
            clues = set()
            for clue in clue_list:
                clues.add(clue["clue"])
            return clues
        except:
            return set()

    def extract_sentences_from_LLM_response(self, resp: str) -> list[str]:
        try:
            clue_list = resp["clues"]
            sentences = []
            for clue in clue_list:
                sentences.append(clue["example_sentences"])
            return sentences
        except:
            return []



    async def get_clues_for_words(self, target_words : str, assassin_word : str) -> list[str]:
        words_str = "[" + ", ".join(target_words) + "]"
        prompt = f"""
        **Objective:**
        You are a linguistic reasoning assistant helping a Codenames AI Codemaster generate smart, safe, and high-utility clues.

        ## OBJECTIVE
        Given the list of target words: [{words_str}], generate **up to 5 distinct one-word clues** that are **strongly related to ALL of the target words**. 
        Each clue should be:
        - A **single English word** (no phrases, plurals, names, or proper nouns).
        - Strongly and clearly semantically related to **all the target words**. Don't try to be too clever - directness is more important. The clearer the clue, the better.
        - **Safe**, meaning the clue must NOT in anyway relate to the dangerous word "{assassin_word}".
        - Think like a human: Your reasoning for choosing each clue must connect strongly to commonsense English knowledge such that an average college graduate (no specific major) can understand it - no extremely niche references (very common topics are allowed).

        ## IMPORTANT: Be as quick as you can.

        ## RULES
        - All clues must be one single English word only.
        - All example sentences must clearly show a natural connection between the clue and each target word. 
        - Do not output anything except the JSON object.

        ## Examples
         - Examples of good clues: 
             + clue "animal" for target words ["salmon", "chicken"].
             + clue "superhero" for target words ["batman", "iron"]. Because obviously batman is a superhero, and "ironman" is a superhero.
             + clue "hogwarts" for the target words ["school", "spell", "lion"].
         - Examples of bad clues:
             + clue "big" for target words ["tower", "london"] and assassin word "stream". Even if you are aiming for the common connection "Tower of big ben in London", the clue "big" is too vague and could potentially lead your guesser to guessing the assassin word "stream" as a stream can also be big.
             + clue "deer" for target words ["buck", "bear", "robin"]. Even though these are all animals, they are not strongly related to each other at all except for "buck" and "deer".  This diverges significantly from how humans often generate and interpret clues for Codenames.
             + clue "mammal" for target words ["walrus", "bear", "eagle"]. As obviously an eagle is not a mammal. Your clue must strongly relate to ALL of the target words. A much safer and stronger clue is "animal".
             + clue “djedkare” for target words "egypt" and "king". Even though it refers to the name of the ruler of Egypt in the 25thcentury B.C., and therefore connects the words “egypt” and “king", it is so niche that it does not reflect the average person’s knowledge of the English language and is likely to yield random guesses if presented to a human player. 

        ## OUTPUT
        Respond only with a valid JSON object in the format:
        {{
          "clues": [
            {{
              "clue": "<one-word clue>",
              "example_sentences": [
                "<Example using clue with target word 1>",
                "<Example using clue with target word 2>",
                "<Example using clue with target word 3>"
              ]
            }},
            ...
          ]
        }}


        """ 
        history = [{"role": "user", "content": prompt}]
        response = await self.client.chat.completions.create(
            messages=history,
            model="gpt-4.1",
            response_format={ "type": "json_object" }
        )
        json_res = self.LLM_response_to_JSON(response.choices[0].message.content)
        # print(self.extract_sentences_from_LLM_response(json_res))
        return self.extract_clues_from_LLM_response(json_res)




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




