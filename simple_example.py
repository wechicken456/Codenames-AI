import json
from game import Game
from players.codemaster_GPT import AICodemaster
from players.guesser_GPT import AIGuesser


def simpleExample():

    print("\nclearing results folder...\n")
    Game.clear_results()

    seed = 0

    print("starting GPT vs. GPT game")
    cmr_kwargs = {"model": "GPT", "version": "gpt-4o-2024-05-13"}
    gr_kwargs = {"model": "GPT", "version": "gpt-4o-2024-05-13"}
    cmb_kwargs = {"model": "GPT", "version": "gpt-4o-2024-05-13"}
    gb_kwargs = {"model": "GPT", "version": "gpt-4o-2024-05-13"}
    Game(AICodemaster, AIGuesser, AICodemaster, AIGuesser, seed=seed, do_print=True, game_name="GPT-GPT",
         cmr_kwargs=cmr_kwargs, gr_kwargs=gr_kwargs, cmb_kwargs=cmb_kwargs, gb_kwargs=gb_kwargs).run()

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
