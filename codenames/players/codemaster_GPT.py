from gpt_manager import game_rules, GPT
from players.codemaster import Codemaster
import re
import time


class AICodemaster(Codemaster):

    def __init__(self, team="Red"):
        super().__init__()
        self.team = team
        self.opponent_team = "Blue" if team == "Red" else "Red"
        # system_prompt = game_rules + "You are playing the game Codenames as the " + team + " Codemaster. \n"
        # system_prompt += "Provide a list of clues in the format of a list of tuples on the same line: (clue1, number1), (clue2, number2), .... "
        # system_prompt += "The maximum number of clues is 5.\n"
        # system_prompt += "Remember that the clue cannot be derived from or derive one of the words on the board. For example, if the word is 'running', the clue cannot be 'run' or 'runner'. "
        # system_prompt += f"A clue must NOT relate to the {self.opponent_team}'s words. "
        # system_prompt += "The clue must be a single English word.\n"
        # system_prompt += "Output the list on the last line at the end of your response."
        system_prompt = """

You are an expert AI Codemaster for the game Codenames.
**Your Task: You will be provided a list of target words and an Assassin word. Your goal is to generate a list of high-quality and **immediately understandable** clues that target all the target words while maximizing clarity and minimizing ambiguity. Your clue also has to absolutely avoid any connnection to the assassin word. You might provide up to 3 clues for this list.**

**Clue Generation Principles:**
1. Clue Rules (Crucial):
    * **Strong Semantic Relation:** Must be **immediately and clearly** semantically related to **all** of your team's intended target words, using *primary meanings* or *widely recognized themes* that a typical human guesser would quickly understand.
    * **Single Word:** Must be a single English word.
    * **No Direct Forms:** Must NOT be a form (inflection, derivation) of any word currently visible on the board (e.g., if 'DRIVE' is on board, 'DRIVER', 'DRIVING', 'DRIVES' are invalid).
    * **No Substring/Superstring Containment (if related):** Must NOT contain a word on the board (e.g., if 'FIRE' is on board, 'FIREMAN' is invalid) or be a substring of another word on the board (e.g., if 'SKYLINE' is on the board, 'SKY' is invalid).

2. **Overall Clue Quality (Reinforcing Clarity):**
    * **Strength & Directness of Association:** The clue must strongly connect to *all* intended target words via their *primary meanings* or *obvious shared themes* (e.g., “fruit” for APPLE, BANANA, ORANGE is clear).
    * **Unambiguity:** Ensure the clue has minimal overlap with the Assassin word. Simulate a human guesser's first thoughts to confirm. 
     * **Think Like a Typical Human Guesser:** The clue must be instantly recognizable and understandable to a typical human with general knowledge (not an AI or expert). Avoid clues that rely on niche, context-specific, or obscure associations (e.g., avoid “hair” for BRUSH, FALL, CYCLE, as “hair cycle” is not a widely recognized phrase.).
    * **Avoid clue words that need to pair with other words to make a phrase**: for example, the clue "hair" that targets "FALL" and "CYCLE" is unacceptable if your reasoning is it's supposed to evoke “hair fall” and “hair cycle". 
    * **Cleverness WITH Clarity:** Creative clues are acceptable only if they remain perfectly clear, safe, and immediately understandable. Obscurity for cleverness is unacceptable.

3. **Safety First (Assassin Avoidance):**
    * **ABSOLUTELY AVOID** any clue that has even a remote chance of relating to any concepts related to he ASSASSIN word. This is the highest priority negative constraint.

* **Examples of Good vs. Bad Clues:**
        * **Good Clue:** “Ocean” for WAVE, TIDE, CORAL (clear, direct, universally understood theme of ocean-related terms).
        * **Bad Clue:** “Hair” for BRUSH, FALL, CYCLE (niche associations like “hair fall” and “hair cycle” are not obvious; a human might think of SCALP or COMB instead).
        * **Bad Clue:** “Revolver” for PISTOL, PLOT, WITCH (connection to PLOT is abstract, and WITCH is unrelated for humans).

**Output Format:**
For each clue, think about why you would choose it, and revisit the guidelines/strategies given to see if the clue matches all the criterias. If you picked a clue and later realized that it's not good, then discard that clue. For each chosen clue, provide reasoning that includes:
* why it relates to the provided list of target words. 
* why it adheres to the guidelines/rules provided.
* Confirmation that the clue avoids the Assassin word. 
Then, on a new line at the end, list the clues as pairs separated by commas. All clues must be on the same line at the end.
EXAMPLE if you choose to provide 3 clues: 
clue1, clue2, clue3

---

**You are the {self.team} Codemaster. Analyze the provided game state and generate your list of candidate clues now.**
"""

        
        print(system_prompt)
        self.manager = GPT(system_prompt=system_prompt, version="o3-mini")

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_remaining_options(self):
        # Converts the words and map variables into a more gpt-friendly text format
        red, blue, civilian, assassin = [], [], [], []
        for i in range(len(self.words)):
            if self.words[i][0] == '*':
                continue
            if self.maps[i] == "Red":
                red.append(self.words[i])
            if self.maps[i] == "Blue":
                blue.append(self.words[i])
            if self.maps[i] == "Civilian":
                civilian.append(self.words[i])
            if self.maps[i] == "Assassin":
                assassin.append(self.words[i])
        return red, blue, civilian, assassin

    def get_clue(self):
        invalid_timer = 0
        clue = None
        number = None
        red, blue, civilian, assassin = self.get_remaining_options()
        prompt = ""
        start = time.time()
        while clue is None or number is None:
            if invalid_timer == 0:
                move_history = ""
                history = self.get_move_history()
                for move in history:
                    if "Codemaster" in move[0]:
                        move_history += move[0] + " gives clue " + move[1] + ", " + str(move[2])
                    else:
                        move_history += move[0] + " gives guess the " + move[2] + " word " + move[1] + " and decided to "
                        if move[3] == True:
                            move_history += "keep guessing"
                        else:
                            move_history += "STOP guessing"
                    move_history += "\n"
                prompt += "The remaining words are: "
                prompt += "Red: " + str(red) + ". "
                prompt += "Blue: " + str(blue) + ". "
                prompt += "Civilian: " + str(civilian) + ". "
                prompt += "Assassin: " + str(assassin) + ". "
                prompt += f"Here is the move history of the game: f{move_history}\n"
                prompt += "**You are the {self.team} Codemaster. Analyze the provided game state and generate your list of candidate clues now. Remember that your clues must adhere to the rules provided in the Game Overview section.**"

            response = self.manager.talk_to_ai(prompt)
            try:
                print(f"\n\n[DEBUG]: {response}\n\n")
                response = response.split("\n")
                response = response[-1]
                print(f"[DEBUG]: {response}\n\n")

                split_input = response.upper().strip().split(",")
                clue = re.sub(r'[^A-Z]', '', split_input[0])
                number = int(re.sub(r'[^0-9]', '', split_input[1]))
                if number < 1:          # check that number provided is greater than 0
                    prompt = "The clue number must be greater than zero. "
                    print("Warning! Invalid clue: " + response + "\n" + prompt)
                    clue = None
                    number = None
                    invalid_timer += 1
                else:                   # check that clue does not derive from any of the remaining words on the board.
                    for i in range(len(self.words)):
                        if self.words[i][0] != '*':
                            if clue in self.words[i] or self.words[i] in clue:
                                prompt = f"The clue cannot be derived from one of the words on the board: your clue = {clue}, word on board = {self.words[i]}.\nRevisit the guidelines for clue validity rules, and provide another clue"
                                print("Warning! Invalid clue: " + response + "\n" + prompt)
                                clue = None
                                number = None
                                invalid_timer += 1 
                                break
                        
            except:
                print("Warning! Invalid clue: " + response +
                      "\nThat clue format is invalid. ")
                clue = None
                number = None
                prompt = "That clue format is invalid. "
                invalid_timer += 1
            if invalid_timer > 10:
                print("You have made too many invalid clues, selecting a default empty clue")
                clue = ""
                number = 1
        print(f"[DEBUG] time to get clue: {time.time() - start}")
        return [clue, number]
