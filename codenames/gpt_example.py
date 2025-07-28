import json
from game import Game
from players.codemaster_GPT import AICodemaster as RedCodemaster
from players.guesser_GPT import AIGuesser as RedGuesser
from players.codemaster_example import AICodemaster as BlueCodemaster
from players.guesser_example import AIGuesser as BlueGuesser
from time import time

def simpleExample(single_team):

    print("\nclearing results folder...\n")
    Game.clear_results()

    seed = time()

    print("starting GPT game")
    Game(RedCodemaster, RedGuesser, BlueCodemaster, BlueGuesser, seed="9876", do_print=True, game_name="GPT-Sample", single_team=single_team).run()

    # display the results
    print(f"\nfor seed {seed} ~")
    with open("results/bot_results_new_style.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            game_name = game_json["game_name"]
            game_time = game_json["time_s"]
            game_score = game_json["total_turns"]
            game_winner = game_json["winner"]
            print(f"time={game_time:.2f}, turns={game_score}, name={game_name}, winner={game_winner}")


if __name__ == "__main__":
    simpleExample(False)    # Two Teams Track
    #simpleExample(True)     # Single Team Track
