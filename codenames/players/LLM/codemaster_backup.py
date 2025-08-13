#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


def format_backup_prompt(our_words : list[str], enemy_words : list[str], civilian_words : list[str], assassin_word: list[str], move_history):
    """Formats the prompt for the LLM Codemaster backup with more elaborate rules."""

    prompt = f"""
You are an expert Codenames AI Codemaster. Your primary clue generation system has failed to find any suitable clues for the current turn. You must now generate a single high-quality clue from scratch based on the provided game state. Your reasoning must be clear, safe, and based on common knowledge.

**Game State:**
- **Your Team's Words:** {our_words}
- **Enemy Words:** {enemy_words}
- **Civilian Words:** {civilian_words}
- **Assassin Word:** {assassin_word}

**Move History:**
{move_history}

---
**PRINCIPLES FOR GENERATING YOUR CLUE**

1.  **Clarity and Commonsense First:** Your reasoning for choosing each clue must connect strongly to commonsense English knowledge. The goal is for an average person to immediately think of the target words when they read your clue.
    - **DO NOT** use niche references. Avoid specialized jargon, academic terms, or deep knowledge of specific history, science, or pop culture. Assume the guesser has general knowledge, but not specific expertise.
    - **Bad Example:** For 'EGYPT' and 'PYRAMID', the clue 'IMHOTEP' is too niche. A much better clue is 'PHARAOH'.
    - **Bad Example:** clue "Yunnan" for target word "China" is bad, as only a few people around the world really know the geography of China. "Shanghai" is a much better clue and a much more well-known name.

2.  **Universal Connection to ALL Targets:** The clue's connection must be strong, direct, and equally applicable to EVERY word in your chosen target group. If the connection to even one word is weak, indirect, or requires mental gymnastics, the entire clue is invalid.
    - **Bad Example:** For ['WHALE', 'BEAR', 'EAGLE'], the clue 'MAMMAL' is bad because an eagle is a bird, not a mammal. A better clue would be 'ANIMAL'.

3.  **Absolute Safety:** The clue must not be related in any way to the enemy words, civilian words, or especially the assassin word. Prioritize safety above all else. If a clue is clever but slightly risky, discard it for a safer one, even if it targets fewer words.

---
**STRICT RULES**
- **ABSOLUTE SAFETY:** Your clue MUST NOT have any connection to the Enemy Words, Civilian Words, or the Assassin Word. A clue that even slightly relates to the assassin word is an instant failure and must be avoided at all costs.
- **SINGLE WORD ONLY:** The clue must be a single English word. No phrases, no hyphens, no proper nouns unless they are extremely common (e.g., 'HOLLYWOOD' or 'HOGWARTS').
- **NO DERIVED OR CONTAINED FORMS:** The clue cannot be a form of, be a part of, be a substring of, be a tense of, or contain any word currently on the board (your words, enemy words, civilian words, or the assassin). This is a strict rule.
    - **Example:** If 'MOON' is on the board, clues like 'MOONLIGHT' or 'HONEYMOON' are illegal.
    - **Example:** If 'DRIVE is on the board, 'DROVE' is illegal.

---
**YOUR TASK**

1.  Analyze **Your Team's Words** to find the best possible group of 2 or more related words.
2.  Following all principles and rules above, generate the single best clue for that group.
3.  Respond ONLY with a valid JSON object containing your chosen clue, the number of words it targets, and the list of words you are targeting.

**Output Format Example:**
{{
  "clue": "FOREST",
  "number": 2,
  "target_words": ["tree", "leaf"]
}}
"""
    return prompt


class LLMBackup:
    def __init__(self, openAI_api_key):
        self.client = AsyncOpenAI(api_key=openAI_api_key)

    async def get_backup_clue(self, our_words : list[str], enemy_words : list[str], civilian_words : list[str], assassin_word : str, move_history):
        """
        Calls the LLM to generate a clue from scratch when the primary system fails.
        """
        prompt = format_backup_prompt(our_words, enemy_words, civilian_words, assassin_word, move_history)

        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4.1", # Or your preferred model
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            clue_data = json.loads(content)

            clue = clue_data.get("clue", "PASS")
            number = clue_data.get("number", 1)
            target_words = clue_data.get("target_words", our_words)

            print(f"[+] LLMBackup gave clue: ({clue.upper()}, {number}, {target_words})")

            if not isinstance(clue, str) or not isinstance(number, int) or number < 1:
                return "PASS", our_words

            return clue.upper(), target_words

        except Exception as e:
            print(f"[!] LLM Backup Codemaster failed: {e}")
            # If the LLM fails completely, give a safe pass clue.
            return "PASS", our_words


