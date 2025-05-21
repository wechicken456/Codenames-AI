from transformers import AutoTokenizer, AutoModelForCausalLM 
from dotenv import load_dotenv
from dotenv import load_dotenv
from huggingface_hub import login
import os 

load_dotenv()
login(token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"))


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


class Qwen3:

    def __init__(self, system_prompt, version):
        super().__init__()
        version = "google/gemma-3-27b-it"
        self.model_version = version 
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModelForCausalLM.from_pretrained(version, device_map=0, torch_dtype="auto")
       
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def talk_to_ai(self, prompt):
        self.conversation_history.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=8192) # outputs dim: (batch_size, input prompt len in tokens + output prompt len in tokens)
        outputs = outputs[0][len(inputs.input_ids[0]):].tolist()   
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(outputs[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(outputs[:index], skip_special_tokens=True).strip("\n")
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
