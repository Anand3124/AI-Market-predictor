"""Command line: python -m connect4 {train,arena,play} ..."""
from __future__ import annotations

import argparse
import os
import random
import sys

from .agents import make_agent
from .arena import arena
from .game import Board
from .model import ValueModel
from .train import train

DEFAULT_MODEL = os.path.join("models", "learned.json")
AGENTS = ("random", "greedy", "learned", "mcts")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1, got %s" % text)
    return value


def load_model(path: str) -> ValueModel:
    if os.path.exists(path):
        return ValueModel.load(path)
    print("no model at %s, using an untrained one (run: python -m connect4 train)" % path)
    return ValueModel()


def cmd_train(args):
    train(args.games, args.model, args.eval_every, args.eval_games, args.epsilon, args.lr,
          args.seed, history_path=os.path.join(os.path.dirname(args.model) or ".", "history.csv"))
    print("saved model to %s" % args.model)


def cmd_arena(args):
    model = load_model(args.model)
    a = make_agent(args.a, model, args.simulations)
    b = make_agent(args.b, model, args.simulations)
    r = arena(a, b, args.games, random.Random(args.seed))
    print("%s vs %s over %d games: %d - %d (%d draws)  ->  %s wins %.0f%%"
          % (args.a, args.b, args.games, r["a"], r["b"], r["draws"], args.a, 100 * r["a"] / args.games))


def cmd_play(args):
    bot = make_agent(args.opponent, load_model(args.model), args.simulations)
    rng = random.Random(args.seed)
    board = Board()
    human = 1 if args.first == "human" else -1
    print("You are %s, the bot is %s. Enter a column number 1-7.\n" % ("X" if human == 1 else "O",
                                                                        "O" if human == 1 else "X"))
    while board.winner is None:
        print(board.render(), "\n")
        if board.player == human:
            col = _ask_move(board)
        else:
            col = bot.choose(board, rng)
            print("bot plays column %d" % (col + 1))
        board.play(col)
    print(board.render(), "\n")
    if board.winner == 0:
        print("draw!")
    elif board.winner == human:
        print("you win!")
    else:
        print("the bot wins!")


def _ask_move(board: Board) -> int:
    while True:
        try:
            raw = input("your move: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if raw.isdecimal() and int(raw) - 1 in board.legal_moves():
            return int(raw) - 1
        print("pick one of: %s" % " ".join(str(c + 1) for c in board.legal_moves()))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m connect4", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="improve the model by self-play")
    t.add_argument("--games", type=positive_int, default=5000)
    t.add_argument("--eval-every", type=positive_int, default=500)
    t.add_argument("--eval-games", type=positive_int, default=100)
    t.add_argument("--epsilon", type=float, default=0.1, help="exploration rate")
    t.add_argument("--lr", type=float, default=0.01, help="learning rate")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("arena", help="pit two agents against each other")
    a.add_argument("a", choices=AGENTS)
    a.add_argument("b", choices=AGENTS)
    a.add_argument("--games", type=positive_int, default=100)
    a.set_defaults(func=cmd_arena)

    g = sub.add_parser("play", help="play against a bot in the terminal")
    g.add_argument("--opponent", choices=AGENTS, default="learned")
    g.add_argument("--first", choices=("human", "bot"), default="human")
    g.set_defaults(func=cmd_play)

    for s in (t, a, g):
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--seed", type=int, default=None)
        s.add_argument("--simulations", type=positive_int, default=200, help="MCTS playouts per move")

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
