#!/usr/bin/env python
# coding: utf-8

# In[34]:


import httpx
import asyncio
import spacy
from dotenv import load_dotenv
import os
from nltk.stem import WordNetLemmatizer
import lemminflect
import time
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nltk.corpus import wordnet as wn

load_dotenv()

try:
    WordNetLemmatizer().lemmatize("test") # Just a test to trigger lookup error if not downloaded
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual Wordnet


# In[35]:


allowed_word_types = [
    "ADJ",	
    "ADV",
    "INTJ",	
    "NOUN",
    "PROPN",	
    "VERB"
]

NOUN_RELS = ["CapableOf", "IsA", "UsedFor", "AtLocation", "HasPrerequisite", "HasProperty", "ReceivesAction", "CreatedBy", "Causes", "HasA", "MadeOf"]
NOUN_REV_RELS = ["AtLocation", "IsA", "PartOf"]
VERB_RELS = ["MannerOf", "HasSubevent", "MotivatedByGoal", "IsA", "HasFirstSubevent", "HasLastSubevent"]
VERB_REV_RELS = ["CapableOf", "MannerOf", "CausesDesire", "UsedFor", "CreatedBy", "ReceivesAction", "HasFirstSubevent"]
ADJ_RELS = ["SimilarTo", "RelatedTo", "HasProperty"] 


stopwords = set([
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 
    'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 
    'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 
    'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 
    'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 
    'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 
    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 
    'here', 'than', 'get', 'put', 'thing', 'something', 'okay', 'stuff', 'things', 'thingy', 'whatever', 'whenever', 'whoever', 'wherever', 'alright', 
    'really', 'yeah', 'nope', 'yep', 'kinda', 'sorta', 'maybe', 'almost', 'like', 'just', 'anyway', 'somehow', 'someone', 'anyone', 'everyone', 'none',
    'somebody', 'nobody', 'everybody', 'thingamajig', 'doohickey', 'meh', 'huh', 'yo', 'hi', 'hello', 'bye', 'goodbye', 'welcome', 'please', 'thanks', 
    'thank', 'ok', 'sure', 'nah', 'uh', 'um', 'oops', 'woo', 'yay', 'aw', 'ugh', 'eh', 'hmm', 'woah', 'uhh', 'lol', 'omg', 'idk', 'btw', 'fyi', 'brb', 
    'irl', 'tbh', 'really', 'literally', 'basically', 'seriously', 'clearly', 'surely', 'maybe', 'perhaps', 'somehow', 'sometime', 'eventually', 
    'already', 'soon', 'later', 'often', 'always', 'never', 'sometimes', 'anyway', 'just', 'even', 'simply', 'only', 'nearly', 'almost', 'barely', 
    'hardly', 'likely', 'possibly', 'probably', 'generally', 'typically', 'essentially', 'kindly', 'mostly', 'virtually', 'supposedly', 'approximately', 
    'theoretically', 'arguably'
])


# In[37]:


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

def get_word_classes(word : str) -> set:
    pos_set = set()
    for synset in wn.synsets(word):
        #print(f"Synset: {synset}")
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

def remove_stopwords(words : str) -> str:
    word_list = words.split()
    filtered_words = [word for word in word_list if word.lower() not in stopwords]
    return " ".join(filtered_words)


class ConceptNet:
    def __init__(self, our_words : list[str], enemy_words : list[str], civilian_words : list[str], assassin_word : str):
        self.our_words = [word.lower() for word in our_words]
        self.enemy_words = [word.lower() for word in enemy_words]
        self.civilian_words = [word.lower() for word in civilian_words]
        self.assassin_word = assassin_word.lower()
        self.all_board_words = set(our_words + enemy_words + civilian_words + [assassin_word])

        self.clue_candidates = defaultdict(set)

        self.strong_relations = {'IsA', 'PartOf', 'HasA', 'UsedFor', 'AtLocation', 'HasProperty', 'MannerOf', 'MotivatedByGoal'}

        # Spacy
        self.nlp = spacy.load("en_core_web_trf")

        self.conceptnet_client = httpx.AsyncClient(http2=True)

        # All forms of board words
        self.lemmas = set()
        for word in self.all_board_words:
            self.lemmas.update(word)
            self.lemmas.update(get_all_inflections(word))
            self.lemmas.update(get_all_possible_lemmas(word))

        # Create graph from ConceptNet queries
        self.graph = nx.DiGraph()


    async def fetch_conceptnet(self, url : str) -> dict:
        r = await self.conceptnet_client.get(url, follow_redirects=True)
        return r.json()

    def _filter(self, words: str) -> list[str]:
        """
        return the filtered list of words. Each word must NOT appear as a lemma or an inflection of any words on the board (which is given by self.lemmas).
        """
        doc = self.nlp(words)
        l = list(filter(lambda token: (token.pos_ in allowed_word_types) and (token.lemma_ not in self.lemmas) and (token.text not in self.lemmas), doc)) 
        # l is a list of tokens, convert them to string to return
        return [token.text for token in l]

    def _process_edge(self, edge : dict, target_word):
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

        start_node_label = remove_stopwords(start_node["label"].lower())
        end_node_label = remove_stopwords(end_node["label"].lower())

        if target_word in start_node_label:
            if end_node["language"] != "en": return []
            label_words = self._filter(end_node_label)
        else:
            if start_node["language"] != "en": return []
            label_words = self._filter(start_node_label)
        return label_words

    async def _fetch_relations(self, target_word: str, rel_list: list, limit : int, is_rev: bool = False) -> set:
        """Generic function to fetch and process edges for a list of relations. Create multiple parallel tasks to process these relations."""
        clues = set()
        tasks = []
        for rel in rel_list:
            node_param = "end" if is_rev else "start"
            url = f"http://api.conceptnet.io/query?{node_param}=/c/en/{target_word}&rel=/r/{rel}&limit={limit}"
            tasks.append(self.conceptnet_client.get(url, follow_redirects=True))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(responses):
            if isinstance(res, Exception):
                # print(f"Warning: Request for {target_word} with relation {rel_list[i]} failed: {res}")
                continue

            try:
                api_res = res.json()
                relation_name = rel_list[i]
                for edge in api_res.get("edges", []):
                    processed_clues = self._process_edge(edge, target_word)
                    for clue in processed_clues:
                        clues.add((clue, relation_name, is_rev))
            except (ValueError, KeyError) as e:
                # print(f"Warning: Could not parse JSON for {target_word} with relation {rel_list[i]}: {e}")
                continue
        return clues


    async def _fetch_clues_for_word(self, word: str, specific_rels: list = None, limit : int = 8) -> dict:
        """
        Fetches all potential clues for a single word. Depending on its POS (Part Of Speech), use the relation list accordingly.
        If specific_rels is provided, use those relations instead.
        """
        clues_by_pos = {'noun': set(), 'verb': set(), 'adj': set()}
        word_classes = get_word_classes(word)

        for pos in word_classes:
            rels, rev_rels = [], []
            if specific_rels:
                rels = specific_rels
                rev_rels = specific_rels
            elif pos == 'noun':
                rels, rev_rels = NOUN_RELS, NOUN_REV_RELS
            elif pos == 'verb':
                rels, rev_rels = VERB_RELS, VERB_REV_RELS
            elif pos == 'adj':
                rels, rev_rels = ADJ_RELS, []

            forward_clues, reverse_clues = await asyncio.gather(
                self._fetch_relations(word, rels, limit, is_rev=False),
                self._fetch_relations(word, rev_rels, limit, is_rev=True)
            )
            clues_by_pos[pos].update(forward_clues)
            clues_by_pos[pos].update(reverse_clues)
        return clues_by_pos

    async def build_clue_graph(self):
        """Builds the clue graph by querying ConceptNet up to 2 levels deep."""     

        total_start = time.time()

        # Add all board words first
        for word in self.our_words: self.graph.add_node(word, type='our_word')
        for word in self.enemy_words: self.graph.add_node(word, type='enemy_word')
        for word in self.civilian_words: self.graph.add_node(word, type='civilian_word')
        self.graph.add_node(self.assassin_word, type='assassin_word')

        # Step 1: populate graph with depth-1 clues
        print("\n--- Building Clue Graph (Depth 1) ---")
        d1_tasks = []
        for word in self.our_words:
            d1_tasks.append(self._fetch_clues_for_word(word, limit=5))

        d1_results = await asyncio.gather(*d1_tasks)

        d1_clues_for_stage2 = []
        for i, clues_by_pos in enumerate(d1_results):
            source_word = self.our_words[i]
            # Check if the word is a landmark
            doc = self.nlp(source_word)
            is_landmark = doc.ents and doc.ents[0].label_ in ["GPE", "LOC"]

            for pos, clue_tuples in clues_by_pos.items():
                for clue, relation, is_rev in clue_tuples:
                    if clue in self.all_board_words: continue

                    if not self.graph.has_edge(source_word, clue):
                        self.graph.add_edge(source_word, clue)
                        self.clue_candidates[source_word].add(clue)
                        if relation in self.strong_relations:
                            d1_clues_for_stage2.append((source_word, clue, pos, relation))

        print(f"Graph after depth 1: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        print(f"Time taken: {time.time() - total_start}")

        print("\n--- Building Clue Graph (Depth 2) ---")
        start_2 = time.time()
        d2_tasks = []
        for _, d1_clue, _, rel in d1_clues_for_stage2:
            d2_tasks.append(self._fetch_clues_for_word(d1_clue, specific_rels=[rel], limit=4))

        d2_results = await asyncio.gather(*d2_tasks)

        for i, clues_by_pos in enumerate(d2_results):
            source_word, d1_clue, pos, _ = d1_clues_for_stage2[i]
            for _, clue_tuples in clues_by_pos.items():
                for d2_clue, d2_relation, is_rev in clue_tuples:
                    if d2_clue in self.all_board_words: continue

                    if not self.graph.has_edge(d1_clue, d2_clue):
                        self.clue_candidates[source_word].add(d2_clue)
                        self.graph.add_edge(d1_clue, d2_clue)


        print(f"Finished stage 2.\nTime taken: {time.time() - start_2}")
        print(f"Graph after depth 2: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

        print("\n--- Finished building graph ---")
        print(f"Total time taken: {time.time() - total_start}")


    def find_candidate_clues(self, target_words: list[str]) -> set:
        """
        Finds candidate clues by intersecting the pre-computed clue lists.
        """
        if not target_words:
            return set()

        # Get the list of set of candidate clues for each target word
        try:
            candidate_sets = [set(self.clue_candidates[word.lower()]) for word in target_words]
        except KeyError:
            return set() # A target word might have no valid clues

        if not candidate_sets:
            return set()

        # Get intersection of all the sets
        intersection = candidate_sets[0].intersection(*candidate_sets[1:])
        return intersection


# In[ ]:





# In[ ]:




