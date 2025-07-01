from gpt_manager import game_rules, GPT
from players.codemaster import Codemaster
import re


class AICodemaster(Codemaster):

    def __init__(self, team="Red"):
        super().__init__()
        self.team = team
        system_prompt = game_rules + "You are playing the game Codenames as the " + team + " Codemaster. "
        self.manager = GPT(system_prompt=system_prompt, version="gpt-4o-2024-05-13")

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
        while clue is None or number is None:
            prompt += "The remaining words are: "
            prompt += "Red: " + str(red) + ". "
            prompt += "Blue: " + str(blue) + ". "
            prompt += "Civilian: " + str(civilian) + ". "
            prompt += "Assassin: " + str(assassin) + ". "
            prompt += "Provide a single word clue and number for the guesser in the following format ('pebble',2). "
            prompt += "The clue cannot be derived from or derive one of the words on the board. "
            prompt += "Stick to this format exactly and provide no additional text. "
            response = self.manager.talk_to_ai(prompt)
            print(response)
            try:
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

        return [clue, number]
