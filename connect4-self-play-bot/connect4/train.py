"""Self-play training with TD(0).

The model plays both sides. After every move the value of the previous board is
nudged towards the value of the new board (or the final result when the game ends).
"""
from __future__ import annotations

import os
import random

from .agents import GreedyAgent, LearningAgent, RandomAgent
from .arena import arena
from .game import Board
from .model import ValueModel


def self_play_game(model: ValueModel, rng: random.Random, epsilon: float, lr: float) -> int:
    agent = LearningAgent(model, epsilon)
    board = Board()
    prev = board.copy()
    while board.winner is None:
        board.play(agent.choose(board, rng))
        model.update(prev, model.value(board), lr)
        prev = board.copy()
    model.games_trained += 1
    return board.winner


def evaluate(model: ValueModel, games: int, rng: random.Random) -> dict:
    agent = LearningAgent(model)
    vs_random = arena(agent, RandomAgent(), games, rng)
    vs_greedy = arena(agent, GreedyAgent(), games, rng)
    return {"vs_random": vs_random["a"] / games, "vs_greedy": vs_greedy["a"] / games}


def train(games: int, model_path: str, eval_every: int = 500, eval_games: int = 100,
          epsilon: float = 0.1, lr: float = 0.01, seed: int = 0, history_path: str = None,
          log=print) -> ValueModel:
    if min(games, eval_every, eval_games) < 1:
        raise ValueError("games, eval_every and eval_games must all be at least 1")
    rng = random.Random(seed)
    model = ValueModel.load(model_path) if os.path.exists(model_path) else ValueModel()
    log("starting from %d games trained, weights %s" % (model.games_trained, _fmt(model.weights)))
    log("%8s %10s %10s   weights" % ("games", "vs_random", "vs_greedy"))
    for g in range(1, games + 1):
        self_play_game(model, rng, epsilon, lr)
        if g % eval_every == 0 or g == games:
            scores = evaluate(model, eval_games, rng)
            log("%8d %9.0f%% %9.0f%%   %s" % (model.games_trained, 100 * scores["vs_random"],
                                              100 * scores["vs_greedy"], _fmt(model.weights)))
            if history_path:
                _append_history(history_path, model.games_trained, scores)
            model.save(model_path)
    return model


def _fmt(weights):
    return "[" + ", ".join("%.2f" % w for w in weights) + "]"


def _append_history(path: str, games: int, scores: dict) -> None:
    new = not os.path.exists(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a") as fh:
        if new:
            fh.write("games,vs_random,vs_greedy\n")
        fh.write("%d,%.3f,%.3f\n" % (games, scores["vs_random"], scores["vs_greedy"]))
