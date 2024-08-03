import json
from game import Game
from players.codemaster_GPT import AICodemaster
from players.guesser_GPT import AIGuesser


def simpleExample():

    print("\nclearing results folder...\n")
    Game.clear_results()

    seed = 0

    print("starting GPT vs. GPT game")
    Game(AICodemaster, AIGuesser, AICodemaster, AIGuesser, seed=seed, do_print=True, game_name="GPT-GPT").run()

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
    simpleExample()
