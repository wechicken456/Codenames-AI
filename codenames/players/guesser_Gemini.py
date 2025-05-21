from gemini_manager import game_rules, Gemini
from players.guesser import Guesser
import random
import time


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
    {{"word": "word2_on_board", "status": "revealed_as_RED"}},
    {{"word": "word3_on_board", "status": "revealed_as_BLUE"}},
    {{"word": "word4_on_board", "status": "unrevealed_as_*"}},
    ... (list of all 25 words and their current revealed status: "unrevealed_as_*", "revealed_as_RED", "revealed_as_BLUE", "revealed_as_CIVILIAN", "revealed_as_ASSASSIN")
  ]
* **Your Team Color:** "{self.team}"
* **{self.team} Team Words Remaining:** {{team_words_left_count}}
* **{self.opponent_team} Team Words Remaining:** {{opponent_words_left_count}}

**Your Task: Provide Your Intended Guesses for This Turn**

Based on the clue "{{clue_word}}" (number: {{clue_number}}) and the current board state, provide an ordered list of maximum 3 words you want to guess for your team ({self.team}).
* List the words in the order you want to guess them.
* You should aim to guess at least 1 word and at most {{clue_number}} + 1 words.
* Your Codemaster is trying to give you clear clues that point to your team's words and avoid others.

**Guiding Principles for Making Your Guesses:**

1.  **Understand the Clue & Codemaster's Intent:**
    * Consider primary meanings, common associations, categories, and potential connotations of "{{clue_word}}".
    * Your Codemaster is trying to give clear, common-knowledge clues. They are *not* trying to trick you but are constrained by other words on the board. They believe there are {{clue_number}} strong links.
2.  **Strongest Associations First:** Identify *unrevealed* words on the board that have the strongest, most direct, and most unambiguous semantic connection to "{{clue_word}}". Prioritize these in your guess order.
3.  **Strategic Use of Clue Number ({{clue_number}}):**
    * Your primary goal is to find {{clue_number}} correct words if the associations are strong and safe.
    * **The extra `+1` Guess:** Only consider making a `+1` guess (guessing {{clue_number}} + 1 words) if:
        * You have already successfully identified {{clue_number}} words for your team with this clue.
        * You are *highly confident* that the additional word is also one of your team's words linked to the current clue and is not a trap.
        * The potential benefit outweighs the risk of ending the turn.
    * **Stopping Early:** It is often strategically sound to guess *fewer* than {{clue_number}} words if:
        * You cannot find {{clue_number}} words with high confidence.
        * The remaining potential words have weak, ambiguous, or risky associations.
        * Protecting a lead or avoiding the assassin is paramount.
        * Do not continue guessing if confidence drops significantly after your initial strong matches.
4.  **Board Awareness & Risk Management:**
    * **Assassin Avoidance:** Be extremely cautious if any potential guess has even a remote plausible connection to concepts often associated with negative outcomes or abrupt endings (common assassin themes). If unsure, err on the side of safety.
    * **Opponent's Words & Neutrals:** Actively consider if a word linked to the clue might *also* be a strong fit for the opponent or a neutral word. Try to select words that are *distinctly* related to the clue in a way that your Codemaster likely intended for *your* team.
    * **Consider Revealed Words:** Revealed words (yours, opponent's, neutral) provide context. They might indicate patterns or rule out certain interpretations of clues.
5.  **Infer Teammate & Opponent Tendencies (Advanced):**
    * Review past clues (yours and opponent's) if available. Do they suggest a particular style of cluing or guessing?
    * If your Codemaster has been giving very literal clues, expect literal. If they've been more abstract, be open to that.
**Output Format:**
From your ordered list of potential guesses, select the most confident word. Output it on a single line at the very end of your response. 
---
"""
        self.manager = Gemini(system_prompt=system_prompt, version="gemini-2.5-flash-preview-05-20")

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
            prompt += f"Here is the move history of the game: f{self.get_move_history_str()}\n"
            prompt += "The following is the Codemaster's clue: (" + str(self.clue) + ", " + str(self.num) + "). "
            prompt += "You have picked " + str(self.guesses) + " words this turn. "
            if self.guesses == self.num:
                prompt += "You can guess 1 more word, but you have already reached the clue number provided by the codemaster. It might be risky to guess once more."
            prompt += "Analyze the remaining options and the state of the game based on the move history, and decide if you want to keep guessing."
            prompt += "Would you like to keep guessing? Answer only 'yes' or 'no'. "
            response = self.manager.talk_to_ai(prompt)
            if "yes" in response.lower():
                guess_again = True
            elif "no" in response.lower():
                guess_again = False
            elif invalid_timer > 10:
                guess_again = False
            else:
                time.sleep(1.0)
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

    def get_move_history_str(self) -> str:
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

    def get_answer(self):
        invalid_timer = 0
        guess = None
        prompt = f'**You are the {self.team} Guesser. The clue is "{self.clue}", and the clue number is {self.num}. Analyze the board and the move history of the game and provide your ordered list of guesses now.**'
        while guess is None:
           
            prompt += "The remaining words are: " + str(self.get_remaining_options()) + ". "
            prompt += f"Here is the move history of the game: f{self.get_move_history_str()}\n"
            prompt += "The following is the Codemaster's clue: (" + str(self.clue) + ", " + str(self.num) + "). "
            prompt += "Using the proided guidelines your system prompt, select one of the remaining words that is most associated with this clue."
            prompt += "Feel free to take time to think and reason. On the last line of your response, output the word you are most confident to guess as a single word."
            print("\n\n" + prompt + "\n\n")
            response = self.manager.talk_to_ai(prompt)
            print(response)
            response = response.split("\n")[-1]
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
                time.sleep(1)
                print("Warning! Invalid guess: " + response)
                prompt = "That was not a valid word. "
                invalid_timer += 1
        self.guesses += 1
        return guess
