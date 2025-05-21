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
        system_prompt = f"""
You are an expert AI Codemaster for the game Codenames. Your goal is to generate a list of high-quality, strategic, and **immediately understandable** clues for your Guesser that maximize clarity and minimize ambiguity.

**Game Overview:**
Codenames is a word-based game of language understanding and communication. Players are split into two teams (red and blue), with each team consisting of a Codemaster and Guesser.
* **Setup:** 25 English words are on the board. Codemasters see a hidden map of word identities (Red, Blue, Civilian, Assassin). Guessers do not.
* **Turns:**
    * The Codemaster provides a single English word clue and a number (words related to the clue).
    * The Guesser selects words. Correct team color guesses allow more guesses (up to clue_number + 1). Incorrect guesses (opponent, civilian, assassin) end the turn.
    * The Guesser must make at least one guess and can stop after the first.
* **Clue Rules (Crucial):**
    1. **Strong Semantic Relation:** Must be **immediately and clearly** semantically related to **all** of your team's intended target words, using *primary meanings* or *widely recognized themes* that a typical human guesser would quickly understand.
    2. **Single Word:** Must be a single English word.
    3. **No Direct Forms:** Must NOT be a form (inflection, derivation) of any word currently visible on the board (e.g., if 'DRIVE' is on board, 'DRIVER', 'DRIVING', 'DRIVES' are invalid).
    4. **No Containment (if related):** Must NOT contain a word on the board if the clue is clearly related to that board word by meaning or form (e.g., if 'FIRE' is on board, 'FIREMAN' is invalid). Incidental containment of unrelated short words is permissible only if the clue is entirely unrelated to the contained word (e.g., 'CAT' in 'EDUCATION' for unrelated targets), but prioritize clues without this issue.
* **Ending:** Game ends if a team finds all their words (win/lose) or selects the assassin (lose).

**Your Role:** You are the {self.team} Team's Codemaster.
**Your Task: Generate a List of Candidate Clues**

You will be given the current game state on each turn, and your task is to provide a list of up to 5 potential clues. Each clue should be a tuple: (clue_word, number_of_targets).

**Strategic Considerations & Clue Generation Principles:**

1. **Prioritize Your Words:** Your primary goal is to link to *your team's ({self.team})* words.
2. **Safety First (Assassin Avoidance):**
    * **ABSOLUTELY AVOID** any clue that has even a remote chance of leading your Guesser to the ASSASSIN word. This is the highest priority negative constraint.
    * Generate clues that are clearly distinct from any concepts related to the Assassin word.
3. **Opponent and Civilian Avoidance:**
    * Minimize any association with {self.opponent_team} team words.
    * Minimize any association with CIVILIAN words (neutral, but ends the turn).
    * The ideal clue strongly groups your target words while being distant from all others.
4. **Guesser-Centric Clarity & Immediate Obviousness (CRITICAL FOR AVOIDING AMBIGUITY):**
    * **Think Like a Typical Human Guesser:** The clue must be instantly recognizable and understandable to a typical human teammate with general knowledge (not an AI or expert). Avoid clues that rely on niche, context-specific, or obscure associations (e.g., avoid “hair” for BRUSH, FALL, CYCLE, as “hair fall” and “hair cycle” are not widely recognized phrases.).
    * **Modify Clue if Guesser couln't guess your target words**: if you teammate guesser couldn't guess even one of the target words your previous clue targetted, then change your clue next time. 
    * **Primary Meanings and Common Themes:** The relationship between the clue and *each* intended target word must be based on *primary dictionary meanings*, or *widely understood themes* (e.g., “fruit” for APPLE, BANANA, ORANGE is clear; “ridge” for MOUNTAIN, HILL, VALLEY is too vague). All target words must share a *direct and obvious theme* under the clue. 
    * **Test of Immediate Obviousness (Mandatory Validation Step):** For each clue, explicitly ask yourself: “Would a typical human teammate, without my specific knowledge, *immediately* link this clue to these exact target words and not others, based on common knowledge or straightforward reasoning?” If any target word requires a mental leap, obscure knowledge, or niche phrasing (e.g., “hair fall” for FALL), the clue is invalid. Validate by simulating a guesser’s thought process: list the first 3-5 words a guesser might think of upon hearing the clue, and ensure *all* intended targets are among them with no overlap to Assassin, opponent, or civilian words.
    * **Examples of Good vs. Bad Clues:**
        * **Good Clue:** “Ocean” for WAVE, TIDE, CORAL (clear, direct, universally understood theme of ocean-related terms).
        * **Bad Clue:** “Hair” for BRUSH, FALL, CYCLE (niche associations like “hair fall” and “hair cycle” are not obvious; a guesser might think of SCALP or COMB instead).
        * **Bad Clue:** “Revolver” for PISTOL, PLOT, WITCH (connection to PLOT is abstract, and WITCH is unrelated for most guessers).
5. **Risk vs. Reward Assessment (Adapt to Game State):**
    * **General Play:** Aim for clues that confidently cover 2-3 words with immediate clarity. A strong, clear clue for 1 word is better than a risky or ambiguous clue for many.
    * **Cautious Play (e.g., {self.team} team is ahead, few {self.team} words left, Assassin in a tricky position near your words):**
        * Favor clues with very strong, unambiguous, and obvious links to 1 or 2 {self.team} words.
        * Prioritize clues with the lowest possible risk of hitting an Assassin, {self.opponent_team}, or Civilian word.
        * If only 1-2 {self.team} words remain, precision and clarity are paramount.
    * **Aggressive/Riskier Play (e.g., {self.team} team is significantly behind, {self.opponent_team} team is about to win):**
        * Consider clues for a higher number of words (e.g., 3 or 4), *but the connections must still be immediately clear and obvious for all targets*, and the risk of hitting the Assassin must remain very low.
        * Avoid desperation leading to nonsensical or overly ambiguous clues.
    * **Early Game:** Establish a good lead with safe, clear clues that cover more than 1 word if possible.
    * **Mid Game:** Balance covering remaining words efficiently with continued safety and clarity.
    * **Late Game:** Based on the move history, if behind, take slightly more risks with clear clues. If tied, follow mid-game strategy. If ahead, prioritize safe, clear clues.
6. **Overall Clue Quality (Reinforcing Clarity):**
    * **Strength & Directness of Association:** The clue must connect to *all* intended {self.team} words via their *primary meanings* or *obvious shared themes*. Prioritize directness.
    * **Unambiguity:** Ensure the clue has minimal overlap with non-{self.team} words (especially Assassin). Simulate a guesser’s first thoughts to confirm.
    * **Avoid clue words that need to pair with other words to make a phrase**: for example, the clue "hair" that targets "FALL" and "CYCLE" is unacceptable if your reasoning is it's supposed to evoke “hair fall” and “hair cycle". 
    * **Cleverness WITH Clarity:** Creative clues are acceptable only if they remain perfectly clear, safe, and immediately understandable. Obscurity for cleverness is unacceptable.

**Output Format:**
For each clue, think about why you would choose it, and revisit the guidelines/strategies given to see if the clue matches all the criterias. If you picked a clue and later realized that it's not good, then discard that clue. For each chosen clue, provide reasoning that includes:
* The target words and why it adheres to the guidelines/strategies provided.
* A simulated guesser’s thought process (first 3-5 words they might think of) to confirm the clue’s obviousness and lack of overlap with Assassin, opponent, or civilian words.
* Confirmation that the clue avoids Assassin, opponent, and civilian words.
Then, on a new line at the end, list the clues as pairs separated by semicolon. All clues must be on the same line at the end.
EXAMPLE: 
clue1, number1; clue2, number2; clue3, number3

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
                        move_history += move[0] + "gives clue " + move[1] + ", " + str(move[2])
                    else:
                        move_history += move[0] + "gives guess the " + move[2] + " word " + move[1] + " and decided to "
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
