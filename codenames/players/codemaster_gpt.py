from gpt_manager import talk_to_ai, game_rules
from players.codemaster import Codemaster


class AICodemaster(Codemaster):

    def __init__(self):
        super().__init__()
        self.cm_wordlist = []
        with open('players/cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_wordlist.append(line.rstrip())
        self.conversation_history = [{"role": "system", "content": game_rules},
                                     {"role": "system", "content": "You are playing the association game 'Codenames' as the red codemaster"}]

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
        clue = None
        number = None
        red, blue, civilian, assassin = self.get_remaining_options()
        while clue is None or number is None:
            prompt = "The remaining words are: \n"
            prompt += "Red: " + str(red) + "\n"
            prompt += "Blue: " + str(blue) + "\n"
            prompt += "Civilian: " + str(civilian) + "\n"
            prompt += "Assassin: " + str(assassin) + "\n"
            prompt += "Provide a single word clue and number for the guesser in the following format ('pebble',2). \n"
            prompt += "Stick to this format exactly and provide no additional text. \n"
            # Bonus Prompts
            prompt += "Your clue should be associated with red words. \n"
            prompt += "Your clue should not be associated with any blue, civilian or assassin words. \n"
            prompt += "Make sure your clue is NOT associated in any way with the assassin word. \n"
            # prompt += "Make sure you are able to provide a justification for your clue. \n"
            # prompt += "Think very carefully about how the guesser may interpret your clue. \n"
            response = talk_to_ai(self.conversation_history, prompt)
            try:
                clue = response.split("'")[1]
                number = int(response.split(",")[1].split(")")[0])
                if len(clue.split()) > 1:
                    print("Warning! Invalid clue: " + response)
                    clue = None
            except:
                print("Warning! Invalid clue: " + response)
                clue = None
                number = None
        return [clue, number]
