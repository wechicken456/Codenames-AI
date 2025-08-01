#!/usr/bin/env python
# coding: utf-8

# In[1]:


import httpx
import asyncio
import spacy
from openai import OpenAI
from dotenv import load_dotenv
import os
from nltk.stem import WordNetLemmatizer
import lemminflect

load_dotenv()
#openAI_api_key = "ENTER YOUR API KEY HERE"
openAI_api_key = os.environ.get("OPENAI_API_KEY")

# Download necessary data for WordNetLemmatizer if we haven't already
try:
    WordNetLemmatizer().lemmatize("test") # Just a test to trigger lookup error if not downloaded
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual Wordnet, often needed for full WordNet functionality


# In[2]:


allowed_word_types = [
    "ADJ",	
    "ADV",
    "INTJ",	
    "NOUN",
    "PROPN",	
    "VERB"
]

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


# In[3]:


lemmatizer = WordNetLemmatizer()
def get_word_lemma(word : str, pos_hint : str = None) -> str:
    """
    Gets the lemma of a single word without sentence context using NLTK's WordNetLemmatizer.
    pos_hint can be 'n' (noun), 'v' (verb), 'a' (adjective), 'r' (adverb).
    Defaults to 'n' if no hint is given.
    """
    return lemmatizer.lemmatize(word.lower(), pos=pos_hint) if pos_hint else lemmatizer.lemmatize(word.lower())

def get_all_possible_lemmas(word: str) -> list[str]:
    res = []
    for hint in ["v", "n", "a", "r", "s"]:
        res.append(get_word_lemma(word, hint))
    return res

def get_all_inflections(word : str) -> list[str]:
    inflections = lemminflect.getAllInflections(word)
    res = []
    for _, i in inflections.items():
        res += list(i)
    return res

class Codemaster:
    def __init__(self, our_words : list[str], enemy_words : list[str], civilian_words : list[str], assassin_word : str, my_team : str):
        self.our_words = our_words
        self.enemy_words = enemy_words
        self.civilian_words = civilian_words
        self.assassin_word = assassin_word
        self.my_team = my_team

        self.nlp = spacy.load("en_core_web_trf")

        self.conceptnet_client = httpx.AsyncClient(http2=True)
        self.openai_client = OpenAI(api_key=openAI_api_key)

        self.lemmas = set()
        all_words = our_words + enemy_words + civilian_words + [assassin_word]
        for word in all_words:
            self.lemmas.update(word)
            self.lemmas.update(get_all_inflections(word))
            self.lemmas.update(get_all_possible_lemmas(word))

        print(self.lemmas)

    async def fetch_conceptnet(self, url : str) -> dict:
        r = await self.conceptnet_client.get(url, follow_redirects=True)
        return r.json()

    def filter(self, words: str) -> list[str]:
        doc = self.nlp(words)
        l = list(filter(lambda token: (token.pos_ in allowed_word_types) and (token.lemma_ not in self.lemmas), doc)) 
        # l is a list of tokens, convert them to string to return
        return [token.text for token in l]

    def process_edge(self, edge : dict, target_word):
        """
        for a given target word, and a given edge it has on ConceptNet, process the node this edge points to:
            + First, the node has to be in English.
            + Second, the node may be multi-word, so we need to process each individual word in it. 
            Keep the open class words (adjective, nouns, verbs, etc) as specified in the universal POS tags: https://universaldependencies.org/u/pos/
            + Then, we need to make sure that each word follows the rules of the game (no subword embedding, etc).
        The `self.filter` function performs the last 2 filters.
        """

        # find whether our target word is at the start node or end node of this edge
        start_node = edge["start"]
        end_node = edge["end"]
        if start_node["label"] == target_word:
            if end_node["language"] != "en":
                return []
            label_words = self.filter(end_node["label"])
        else:
            if start_node["language"] != "en":
                return []
            label_words = self.filter(start_node["label"])
        print(label_words)
        return label_words

    def get_word_classes(self, word: str) -> list:
        return []

    async def fetch_and_extract_clues_for_word(self, word : str) -> list:
        """
        For each of our target word, fetch the ConceptNet API for related nodes and process them as potential clues for that target word.
        """
        word_classes = self.get_word_classes(word)

        url = f"http://api.conceptnet.io/c/en/{word}?limit=200"
        api_res = await self.fetch_conceptnet(url)
        edges = api_res["edges"]
        clues = set()
        for edge in edges:
            clues.update(self.process_edge(edge, word))
        return clues

    async def get_all_potential_clues(self):
        """
        Create multiple tasks to asynchronously get clues for each of our target words.
        """
        self.conceptnet_client = httpx.AsyncClient(http2=True)
        tasks = []
        for word in self.our_words:
            tasks.append(asyncio.create_task(self.fetch_and_extract_clues_for_word(word)))
        res = await asyncio.gather(*tasks)
        await self.conceptnet_client.aclose()


# In[ ]:





# In[4]:


blue_words = ["dwarf", "foot", "moon", "star", "ghost", "beijing", "fighter", "roulette", "alps"]
#red_words = ["club", "superhero", "mount", "bomb", "knife", "belt", "robot", "rock", "bar", "lab"]
red_words = ["drive"]
civilian_words = ["dead"]
assassin_word = "agent"
master = Codemaster(red_words, blue_words, civilian_words, assassin_word, "red")
await master.get_all_potential_clues()


# In[ ]:





# In[ ]:





# In[181]:


l = ['drive', 'dwarf', 'foot', 'moon', 'star', 'ghost', 'beijing', 'fighter', 'roulette', 'alp', 'dead', 'agent']
nlp = spacy.load("en_core_web_trf")
doc = nlp("driving")
for token in doc:
    if token.lemma_ not in l:
        print(token.text, token.lemma_)


# In[164]:


response = openai_client.responses.create(
    model="gpt-4.1",
    input = f"""The following are possible word classes: [ADV (adverb), ADJ (adjective), INTJ (interjection), NOUN (noun), PROPN (proper noun), VERB (verb)].

    Determine which of these classes the word 'table' belongs to based on its possible usages in English (and English only).

    At the end of your response, list the corresponding abbreviations (e.g., NOUN, VERB), separated by commas, on a single line.
    """

).output_text.split("\n")[-1]


# In[165]:


response


# In[6]:


client = OpenAI(api_key=openAI_api_key)
system_prompt = """
You are an expert English linguistic. You
"""


response = client.responses.create(
    model="gpt-4.1",
    input = f"""
    The following are possible word classes: [ADV (adverb), ADJ (adjective), INTJ (interjection), NOUN (noun), PROPN (proper noun), VERB (verb)].

    Determine which of these classes the word 'table' belongs to based on its possible usages in English (and English only).

    At the end of your response, list the corresponding abbreviations (e.g., NOUN, VERB), separated by commas, on a single line.
    """

).output_text.split("\n")[-1]

print(response)


# In[8]:


from datasets import load_dataset

load_dataset("olm/wikipedia", language="en", date="20220920")


# In[ ]:




