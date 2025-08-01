# Description

An AI system for playing Codenames that combines multiple language understanding approaches to generate optimal clues and make strategic guesses. The project leverages ConceptNet's semantic knowledge graph for relationship extraction, OpenAI's GPT models, and custom Wikipedia-trained word embeddings using Qwen contextual representations averaged across all contexts.

**PROGRESS**: almost finished Codemaster, still waiting on my custom word embeddings model (qwen) training to complete.

# Directory structure

- [conceptnet](./conceptnet/): All functions related to querying ConceptNet API. Doesn't rank any words as it's the Codemaster.
- [preprocess](./preprocess/): Clean the huggingface wikipedia dump dataset.
- [embs](./embs/): All the `embs*.py` files are used to train on different chunks of the cleaned wikipedia dataset. They're run concurrently on a SLURM cluster. `combine_embs.py` is used to combine all the embeddings across different chunks together.
- [codenames](./codenames/): contains some files that are prototypes of the codemaster/guesser. The rest is in the same directory structure as described in the original repo or in the [GAME.md](./GAME.md) file.
[datasets](./datasets/): contain miscallaneous datasets.




# Codemaster 

- ConceptNet API: used to query for relationships of a word. I found that this is most of the time better than the clues that we get from just prompting LLMs alone.
- Custom Embedding: Distributed pre-processing of Wikipedia dumps and training on them to create global word representations from qwen contextual embeddings.
- Annoy Index: create an [annoy](https://github.com/spotify/annoy) index to enable nearest neighbors retrieval (only used for 1 team version).
- Fast: due to the 60-s limit for the model to give a clue, async requests are made to OpenAI API and ConceptNet API to get clues for the board words simultaneously. 

## Generating clues:
Combine all of these together we have a very diverse set of clues for each target team word. 

1. For each target word group, we find the intersection of the set of clues of the words in this group, and rank them by cosine similarity. 

2. Then, we take the top 3 scores, and combine them with other groups by inserting into a list of tuples like (score, clue word, [target word group list]). Then, sort this list by score.

3. Take the top 5 from this sorted list and prompt the LLM with the game rules and strategies and ask it to pick 1 of the clues.




