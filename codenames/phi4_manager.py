from transformers import AutoTokenizer, AutoModelForCausalLM 
from dotenv import load_dotenv
import os

# https://czechgames.com/files/rules/codenames-rules-en.pdf
# Codemaster = Spymaster, Guesser = Field Operative
game_rules = """
Codenames is a word-based game of language understanding and communication.
Players are split into two teams (red and blue), with each team consisting of a Codemaster and Guesser.
Setup:
At the start of the game, the board consists of 25 English words.
The Codemasters on each team has access to a hidden map that tells them the identity of all of the words (Red, Blue, Civilian or Assassin).
The Guessers on each team do not have access to this map, and so do not know the identity of any words.
Players need to work as a team to select their words as quickly as possible, while minimizing the number of incorrect guesses.
Turns:
At the start of each team's turn, the Codemaster supplies a clue and a number (the number of words related to that clue).
The clue must:
- Be semantically related to the words the Codemaster wants their Guesser to guess.
- Be a single English word.
- NOT be derived from or derive one of the words on the board.
The Guesser then selects from the remaining words on he board, based on the which words are most associated with the Codemaster's clue.
The identity of the selected word is then revealed to all players.
If the Guesser selected a word that is their team's colour, then they may get to pick another word.
The Guesser must always make at least one guess each turn, and can guess up to one word more than the number provided in the Codemaster's clue.
If a Guesser selects a word that is not their team's colour, their turn ends.
The Guesser can choose to stop selecting words (ending their turn) any time after the first guess.
Ending:
Play proceeds, passing back and forth, until one of three outcomes is achieved:
All of the words of your team's colour have been selected -- you win
All of the words of the other team's colour have been selected -- you lose
You select the assassin tile -- you lose

"""


class Phi4:

    def __init__(self, system_prompt, version):
        super().__init__()
        version = "microsoft/Phi-4-reasoning-plus"
        self.model_version = version 
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModelForCausalLM.from_pretrained(version, device_map="auto", torch_dtype="auto")
       
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def talk_to_ai(self, prompt):
        self.conversation_history.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(self.conversation_history, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(inputs.to(self.model.device),
                                 max_new_tokens=4096,
                                 temperature=0.8,
                                 top_k=50,
                                 top_p=0.95,
                                 do_sample=True,)
        response = self.tokenizer.decode(outputs[0])
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
