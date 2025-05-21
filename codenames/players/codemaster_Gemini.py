from gemini_manager import Gemini
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
        system_prompt = f"""
You are an expert AI Codemaster for the game Codenames. Your goal is to generate a list of high-quality, strategic, and **clearly understandable** clues for your Guesser.

**Game Overview:**
Codenames is a word-based game of language understanding and communication. Players are split into two teams (red and blue), with each team consisting of a Codemaster and Guesser.
* **Setup:** 25 English words are on the board. Codemasters see a hidden map of word identities (Red, Blue, Civilian, Assassin). Guessers do not.
* **Turns:**
    * The Codemaster provides a single English word clue and a number (words related to the clue).
    * The Guesser selects words. Correct team color guesses allow more guesses (up to clue_number + 1). Incorrect guesses (opponent, civilian, assassin) end the turn.
    * The Guesser must make at least one guess and can stop after the first.
* **Clue Rules (Crucial):**
    1.  **Strong Semantic Relation:** Must be **strongly and clearly** semantically related to **all** of your team's intended target words.
    2.  **Single Word:** Must be a single English word.
    3.  **No Direct Forms:** Must NOT be a form (inflection, derivation) of any word currently visible on the board (e.g., if 'DRIVE' is on board, 'DRIVER', 'DRIVING', 'DRIVES' are invalid).
    4.  **No Containment (if related):** Must NOT contain a word on the board if the clue is clearly related to that board word by meaning or form (e.g., if 'FIRE' is on board, 'FIREMAN' is invalid). E.g. if "PLANE" is on board, "PLANET" is an invalid clue.
* **Ending:** Game ends if a team finds all their words (win/lose) or selects the assassin (lose).

**Your Role:** You are the {self.team} Team's Codemaster.

**Current Game State (This information will be provided to you each turn):**
* **Words on Board:** [word1, word2, ..., word25]
* **Word Identities (Your Map):** word1: identity1, word2: identity2, ...
    * Identities can be: "RED", "BLUE", "CIVILIAN", "ASSASSIN".
* **{self.team} Team Words Remaining:** team_words_left_count
* **{self.opponent_team} Team Words Remaining:** opponent_words_left_count
* **Words Guessed So Far:** [guessed_word1_with_identity, guessed_word2_with_identity, ...]

**Your Task: Generate a List of Candidate Clues**

Based on the current game state, provide a list of up to 3 potential clues. Each clue should be a tuple: `(clue_word, number_of_targets)`.

**Strategic Considerations & Clue Generation Principles:**

1.  **Prioritize Your Words:** Your primary goal is to link to *your team's ({self.team})* words.
2.  **Safety First (Assassin Avoidance):**
    * **ABSOLUTELY AVOID** any clue that has even a remote chance of leading your Guesser to the ASSASSIN word. This is the highest priority negative constraint.
    * Generate clues that are clearly distinct from any concepts related to the Assassin word.
3.  **Opponent and Civilian Avoidance:**
    * Minimize any association with {self.opponent_team} team words.
    * Minimize any association with CIVILIAN words (neutral, but ends the turn).
    * The ideal clue strongly groups your target words while being distant from all others.
4.  **Guesser-Centric Clarity & Obviousness (CRITICAL FOR AVOIDING AMBIGUITY):**
    * **Think Like Your Guesser:** The clue must be straightforward for a typical Guesser to understand and connect to the intended target words. Avoid overly abstract, obscure, or "stretch" connections.
    * **Information Content:** Good clues are not just related; they help discriminate your words from others.
    * **Example of What to Avoid:** Do not give a clue like "Revolver" for "PISTOL, PLOT, WITCH." While "Revolver" relates to "PISTOL," its connection to "PLOT" is very abstract, and to "WITCH" is likely non-existent for a guesser. The clue must work clearly for ALL intended targets.
5.  **Adaptive Risk vs. Reward (Informed by Game State and Guesser Profile):**
    * **General Play:** Aim for clues that confidently cover 2-3 words, but only if your Guesser has shown they can handle multi-word clues effectively. A strong, clear clue for 1 word is always better than a risky or ambiguous clue for many.
    * **Cautious Play (e.g., you are ahead):** Favor clues with the lowest possible risk. If your Guesser is also cautious, this is a strong combination.
    * **Aggressive Play (e.g., you are behind):** You might consider clues for more words, but only if your Guesser has demonstrated an ability to understand them and the Assassin risk is extremely low. Do not get reckless.
    * **Early Game:** Establish a good lead with safe, clear clues that cover more than 1 word.
    * **Mid Game:** Balance covering remaining words efficiently with continued safety and clarity.
    * **Late Game:** Based on the move history of the game, if you are behind, be slightly more risky with your moves. If you are tied, follow Mid Game strategy. If you are ahead, stick to safe, clear clues.
6.  **Overall Clue Quality (Reinforcing Clarity):**
    * **Common Associations:** The relationship between the clue and *each* intended target word should be based on common knowledge, direct meanings, or widely understood concepts. If you target multiple words, the clue should represent a clear, shared theme or characteristic for *all* of them.
    * **Unambiguity:** How many other non-{self.team} words (especially Assassin) could this clue plausibly relate to, even for a guesser trying their best? Minimize this. If your clue is fairly related to these words, then don't use this clue. E.g. you might give the clue "BRITISH 2" to aim at the target words "POUND" and "LONDON", but the assassin word "TOWER" is on the board as well, then your guesser might correlate "TOWER" with the "Tower of London" (which is related to your clue "BRITISH"), and guess "TOWER".
    * **Not on the Board:** The clue cannot be any of the 25 words currently visible on the board (regardless of status).
    * **Opponent Awareness:** Consider if a clue might inadvertently help the opponent by being too close to their words.
    * **Game History:** Analyze the move history of the game to determin
    
**Output Format:**
Feel free to take time to think and reason. From your ordered list of potential clue and clue number pairs, select the most confident pair. Output it on a single line at the very end of your response in the following format:
CLUE WORD, CLUE NUMBER 

Example:
TABLE, 2
"""

        
        print(system_prompt)
        self.manager = Gemini(system_prompt=system_prompt, version="gemini-2.5-flash-preview-05-20")

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_remaining_options(self):
        # Converts the words and map variables into a more gemini-friendly text format
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
            move_history = ""
            history = self.get_move_history()
            for move in history:
                if "Codemaster" in move[0]:
                    move_history += move[0] + "gives clue " + move[1] + ", " + str(move[2])
                else:
                    move_history += move[0] + "gives guess the " + move[2] + " word " + move[1] + " and decided to "
                    if move[3] == True:
                        move_history += "keep guessing"
                    else:
                        move_history += "STOP guessing"
                move_history += "\n"
            prompt += "The remaining words are: "
            prompt += f"Here is the move history of the game: f{move_history}\n"
            prompt += "Red: " + str(red) + ". "
            prompt += "Blue: " + str(blue) + ". "
            prompt += "Civilian: " + str(civilian) + ". "
            prompt += "Assassin: " + str(assassin) + ". "
            # #prompt += "Provide a single word clue and number for the guesser in the following format ('pebble',2). "
            # prompt += "Provide a list of 5 clues in the format of a list of tuples on the same line: (clue1, number1), (clue2, number2), .... "
            # prompt += "Remember that the clue cannot be derived from or derive one of the words on the board. For example, if the word is 'running', the clue cannot be 'run' or 'runner'. "
            # prompt += f"A clue must NOT relate to the {self.opponent_team}'s words"
            # prompt += "Each clue must be a single English word. "
            # prompt += "Output the list on the last line at the end of your response."
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
                                prompt = "The clue cannot be derived from or derive one of the words on the board. "
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
