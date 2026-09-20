"""Play two agents against each other and count the results."""
from __future__ import annotations

import random

from .agents import Agent
from .game import Board


def play_game(first: Agent, second: Agent, rng: random.Random) -> int:
    """Returns 1 if `first` wins, -1 if `second` wins, 0 for a draw."""
    board = Board()
    players = {1: first, -1: second}
    while board.winner is None:
        board.play(players[board.player].choose(board, rng))
    return board.winner


def arena(a: Agent, b: Agent, games: int, rng: random.Random) -> dict:
    """Agents alternate who moves first. Returns wins for a, wins for b, draws."""
    wins_a = wins_b = draws = 0
    for i in range(games):
        if i % 2 == 0:
            result = play_game(a, b, rng)
        else:
            result = -play_game(b, a, rng)
        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        else:
            draws += 1
    return {"a": wins_a, "b": wins_b, "draws": draws, "games": games}
