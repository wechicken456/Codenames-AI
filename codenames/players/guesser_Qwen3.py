from qwen3_manager import game_rules, Qwen3
from players.guesser import Guesser
import random



class AIGuesser(Guesser):

    def __init__(self, team="Red"):
        super().__init__()
        self.team = team
        self.opponent_team = "Blue" if team == "Red" else "Red"
        self.num = 0
        self.guesses = 0
        # system_prompt = game_rules + "You are playing the game Codenames as the " + team + " Guesser. "
        system_prompt = f"""
You are an expert AI Guesser for the game Codenames. Your goal is to accurately interpret your Codemaster's clue and select your team's words from the board.

**Game Overview (Guesser's Perspective):**
Codenames is a word-based game. You will receive a one-word clue and a number from your Codemaster.
* **Your Task:** Based on this clue and number, you must choose words from the 25 words on the board that you believe are your team's words related to the clue.
* **Board:** Initially, all 25 words are unknown. As words are guessed, their identity (Red, Blue, Civilian, or Assassin) is revealed.
* **Guessing Rules:**
    1.  You must make at least one guess per turn.
    2.  If you guess a word of your team's color, you *may* make another guess.
    3.  You can guess up to one more time than the number your Codemaster gave (e.g., if the number is 2, you can make up to 3 guesses, provided the first 2 are correct).
    4.  If you guess a word belonging to the opponent team, a civilian word, or the assassin word, your turn ends immediately.
    5.  You can choose to stop guessing at any point after your first guess, even if you could make more.
* **Objective:** Help your team identify all its words before the opponent team does, and critically, AVOID picking the Assassin.

**Your Role:** You are the {self.team} Team's Guesser.

**Information You Will Receive This Turn:**
* **Clue Word:** "{{clue_word}}"
* **Clue Number:** {{clue_number}} (This is the number of words your Codemaster intends for you to find with this clue.)
* **Words on Board & Their Status:** [
    {{"word": "word1_on_board", "status": "unrevealed"}},
    {{"word": "word2_on_board", "status": "revealed_as_RED"}},
    {{"word": "word3_on_board", "status": "revealed_as_BLUE"}},
    {{"word": "word4_on_board", "status": "unrevealed"}},
    ... (list of all 25 words and their current revealed status: "unrevealed", "revealed_as_RED", "revealed_as_BLUE", "revealed_as_CIVILIAN", "revealed_as_ASSASSIN")
  ]
* **Your Team Color:** "{self.team}"
* **{self.team} Team Words Remaining:** {{team_words_left_count}}
* **{self.opponent_team} Team Words Remaining:** {{opponent_words_left_count}}

**Your Task: Provide Your Intended Guesses for This Turn**

Based on the clue "{{clue_word}}" (number: {{clue_number}}) and the current board state, provide an ordered list of the words you want to guess for your team ({self.team}).
* List the words in the order you want to guess them.
* You should aim to guess at least 1 word and at most {{clue_number}} + 1 words.
* Your Codemaster is trying to give you clear clues that point to your team's words and avoid others.

**Guiding Principles for Making Your Guesses:**

1.  **Understand the Clue:** Consider the primary meanings, common associations, categories, and potential connotations of the clue word "{{clue_word}}".
2.  **Strongest Associations First:** Identify the *unrevealed* words on the board that have the strongest, most direct, and most obvious semantic connection to the clue word. Prioritize these.
3.  **Consider the Clue Number ({{clue_number}}):**
    * Your Codemaster believes there are {{clue_number}} words strongly related to the clue. Try to identify this many if the associations are clear.
    * If you find {{clue_number}} words that are very strong matches, carefully consider if there's an additional word (the `+1` guess) that is also a very strong and safe match. Do not take the `+1` guess lightly; only do so if you are highly confident.
4.  **Focus on Common Knowledge:** Your Codemaster was instructed to use clues that are relatively easy to understand and based on common associations. Interpret the clue from this perspective. Avoid overly obscure or convoluted interpretations.
5.  **Assess Confidence:** For each potential guess, assess your confidence. If, after identifying some strong matches, the remaining unrevealed words have only weak or tenuous connections to the clue, it's often wise to be conservative and not guess further, even if you haven't reached {{clue_number}} guesses.
6.  **Avoid Speculation:** Do not guess randomly. Each guess should be justifiable based on a clear perceived link to the clue.
7.  **Safety:** While you don't know the identities, remember that incorrect guesses are costly. This reinforces the need for strong associations.

**Output Format:**
Provide your ordered list of guesses as a Python-style list of strings, all on a single line at the very end of your response.
If you decide to guess only one word, provide a list with one word. If you decide to guess multiple, list them in order.
Example: `["WORD_A", "WORD_B"]`

---

**You are the {self.team} Guesser. The clue is "{{clue_word}}" for {{clue_number}}. Analyze the board and provide your ordered list of guesses now.**
"""
        self.manager = Qwen3(system_prompt=system_prompt, version="")

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        li = [clue, num]
        return li

    def keep_guessing(self):
        invalid_timer = 0
        response = None
        prompt = ""
        guess_again = False
        while response is None and self.guesses <= self.num:
            prompt += "The remaining words are: " + str(self.get_remaining_options()) + ". "
            prompt += "The following is the Codemaster's clue: (" + str(self.clue) + ", " + str(self.num) + "). "
            prompt += "You have picked " + str(self.guesses) + " words this turn. "
            prompt += "Would you like to keep guessing? Answer only 'yes' or 'no'. "
            response = self.manager.talk_to_ai(prompt)
            if "yes" in response.lower():
                guess_again = True
            elif "no" in response.lower():
                guess_again = False
            elif invalid_timer > 10:
                guess_again = False
            else:
                response = None
                invalid_timer += 1
                prompt = "That was not a valid response, respond with only 'yes' or 'no'. "
        return guess_again

    def get_remaining_options(self):
        remaining_options = []
        for i in range(len(self.words)):
            if self.words[i][0] == '*':
                continue
            remaining_options.append(self.words[i])
        return remaining_options

    def get_answer(self):
        invalid_timer = 0
        guess = None
        prompt = ""
        while guess is None:
            prompt += "The remaining words are: " + str(self.get_remaining_options()) + ". "
            prompt += "The following is the Codemaster's clue: (" + str(self.clue) + ", " + str(self.num) + "). "
            prompt += "Select one of the remaining words that is most associated with this clue. "
            prompt += "You must select one of the remaining words and provide no additional text.  "
            print("\n\n" + prompt + "\n\n")
            response = self.manager.talk_to_ai(prompt)
            print(response)
            response = response.upper().strip()
            if response in self.words:
                guess = response
            elif response.split(" ")[0].strip() in self.words:
                guess = response.split(" ")[0].strip()
            elif response.split('"')[1:2] in self.words:
                guess = response.split('"')[1:2]
            elif response.split("'")[1:2] in self.words:
                guess = response.split("'")[1:2]
            elif invalid_timer > 10:
                print("You have made too many invalid guesses, selecting random remaining word")
                guess = random.choice(self.get_remaining_options())
            else:
                print("Warning! Invalid guess: " + response)
                prompt = "That was not a valid word. "
                invalid_timer += 1
        self.guesses += 1
        return guess
