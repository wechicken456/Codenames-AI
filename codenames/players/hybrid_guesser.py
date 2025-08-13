#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from sentence_transformers import SentenceTransformer
import json
import asyncio
from nltk.corpus import wordnet as wn

from .guesser import Guesser
from .LLM.codemaster import LLM
from .LLM.codemaster_2 import LLM2
from .conceptnet.conceptnet import ConceptNet
from .annoy_index.annoy_index import Annoy

from nltk.stem import WordNetLemmatizer
import lemminflect
import json
import spacy
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


# In[12]:


load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
nlp = spacy.load("en_core_web_trf")


# In[13]:


def format_llm_guesser_prompt(clue: str, num_guesses: int, remaining_words: list[str]) -> str:
    """Formats the prompt for the LLM guesser fallback."""

    words_str = ", ".join(remaining_words)
    prompt = f"""
You are an expert Codenames player. Your codemaster has given you a clue. Your task is to determine the most likely target words from the list of words remaining on the board.

**Game State:**
- **Clue:** "{clue}"
- **Clue number (number of words to guess):** {num_guesses}
- **Remaining words on board:** [{words_str}]

**Your Task:**
1.  Analyze the clue in the context of all the words on the board.
2.  Identify the {num_guesses} words that are the strongest and most direct targets for the clue "{clue}". Assume that clue was given based on commensense knowledge in the English language. 
3.  Consider extremely common knowledge, wordplay, and semantic relationships.
4.  Return your answer as an ordered list, with the most confident guess first.

**Output Format:**
Respond ONLY with a valid JSON object. The object should contain a single key, "guesses", which holds a list of your chosen words in order.

**Example:**
If the clue is "ANIMAL 2" and you choose "LION" and "TIGER", your output should be:
{{
  "guesses": ["LION", "TIGER"]
}}

"""
    return prompt


class GPTManager():
    def __init__(self, api_key):
        self.openai_client = OpenAI(api_key=api_key)
        self.llm_conversation_history = []

    def talk_to_ai(self, prompt):
        self.llm_conversation_history.append({"role": "user", "content": prompt})
        response = self.openai_client.chat.completions.create(
            messages=self.llm_conversation_history,
            model="gpt-4.1",
            response_format={ "type": "json_object" }
        )
        response = response.choices[0].message.content
        self.llm_conversation_history.append({"role": "assistant", "content": response})
        return response


    def reset_LLM_conversation_history(self):
        self.llm_conversation_history = []


class AIGuesser(Guesser):
    def __init__(self, team="Red"):
        super().__init__()
        cwd = os.path.dirname(__file__) 
        annoy_path = os.path.join(cwd, "annoy_index")
        self.emb = Annoy(annoy_path=annoy_path)
        self.gpt_manager = GPTManager(api_key=openai_api_key)

        # Game state
        self.team = team
        self.board_words = []
        self.clue = None
        self.num_guesses = 0

        # Internal strategy variables
        self.planned_guesses = []
        self.turn_in_progress = False

        self.w_fitness = 0.8
        self.w_cohesion = 0.2
        self.CONFIDENCE_THRESHOLD = 0.35 # minimum score for subset to be considered to use as guesses 

    def set_board(self, words_on_board: list[str]):
        self.board_words = [word.lower() for word in words_on_board]

    def set_clue(self, clue: str, num_guesses: int):
        self.clue = clue.lower()
        self.num_guesses = num_guesses
        self.turn_in_progress = True
        self.planned_guesses = []

    def get_answer(self) -> str:
        if not self.planned_guesses:
            remaining_words = [word for word in self.board_words if '*' not in word]
            self._formulate_plan(remaining_words)

        next_guess = self.planned_guesses.pop(0)
        if not self.planned_guesses:
            self.turn_in_progress = False
        print(f"Guesser giving guess {next_guess.upper()}")
        return next_guess.upper()            

    def keep_guessing(self) -> bool:
        return self.turn_in_progress and len(self.planned_guesses) > 0

    def _score_subset(self, subset: tuple[str], clue_word_sims: dict[str, float]) -> float:
        fitness_score = np.mean([clue_word_sims[word] for word in subset])
        cohesion_score = self.average_pairwise_similarity(list(subset))
        return float((self.w_fitness * fitness_score) + (self.w_cohesion * cohesion_score))

    def _formulate_plan(self, remaining_words: list[str]):
        if not self.clue or self.num_guesses == 0:
            self.planned_guesses = []
            return

        clue_emb = self.emb.encode([self.clue])[0]
        remaining_embs = self.emb.encode(remaining_words)
        sims = cosine_similarity([clue_emb], remaining_embs)[0]
        clue_word_sims = {word: sim for word, sim in zip(remaining_words, sims)}

        num_to_find = min(self.num_guesses, len(remaining_words))
        if num_to_find == 0:
            self.planned_guesses = []; return

        best_subset = None
        max_score = -np.inf

        if num_to_find == 1:
            best_word = max(clue_word_sims, key=clue_word_sims.get)
            best_subset = (best_word,)
            # Score for a single word is just its similarity to the clue.
            max_score = clue_word_sims[best_word]
        else:
            candidate_subsets = combinations(remaining_words, num_to_find)
            for subset in candidate_subsets:
                score = self._score_subset(subset, clue_word_sims)
                if score > max_score:
                    max_score = score
                    best_subset = subset

        # Confidence check. Fall back to asking LLM to select words.
        if max_score < self.CONFIDENCE_THRESHOLD or best_subset is None:
            print(f"[!] Low confidence (score: {max_score:.3f}). Falling back to LLM...")
            self.planned_guesses = self._get_guesses_from_llm(remaining_words)
        else:
            ordered_guesses = sorted(list(best_subset), key=lambda w: clue_word_sims[w], reverse=True)
            self.planned_guesses = ordered_guesses

    def _get_guesses_from_llm(self, remaining_words: list[str]) -> list[str]:
        """Calls the LLM to get guesses when the vector model is not confident."""
        try:
            prompt = format_llm_guesser_prompt(self.clue, self.num_guesses, remaining_words)
            response_str = self.gpt_manager.talk_to_ai(prompt)
            response_json = json.loads(response_str)
            guesses = response_json.get("guesses", [])
            valid_guesses = [g.lower() for g in guesses if g.lower() in remaining_words]
            return valid_guesses[:self.num_guesses]
        except Exception as e:
            print(f"[ERROR] LLM Fallback: {e}")
            # More fallback: just guess the most similar words if LLM fails
            return sorted(remaining_words, key=lambda w: cosine_similarity(self.emb.encode([self.clue]), self.emb.encode([w]))[0][0], reverse=True)[:self.num_guesses]


    def average_pairwise_similarity(self, words: list[str]) -> float:
        if len(words) < 2: 
            return 1.0 
        embeddings = self.emb.encode(words)
        sim_matrix = cosine_similarity(embeddings)
        triu_indices = np.triu_indices(len(words), k=1)
        pairwise_sims = sim_matrix[triu_indices]
        return float(np.mean(pairwise_sims)) if pairwise_sims.size > 0 else 0.0


# In[ ]:


# In[ ]:





# In[ ]:




