"""A tiny value function: V(board) = tanh(w . features), learned by TD(0).

Features are counted from player 1's point of view, so V near +1 means
player 1 is winning and V near -1 means player -1 is winning.
"""
from __future__ import annotations

import json
import math
import os

from .game import CENTER, ROWS, WINDOWS, Board

FEATURE_NAMES = ("singles", "pairs", "triples", "center")
SCALE = (10.0, 5.0, 2.0, 3.0)  # keeps w . f in a range where tanh is not saturated


def features(board: Board):
    """[singles, pairs, triples, center]: each is (player 1 count - player -1 count).

    A window counts for a player only if the other player has no piece in it,
    because a mixed window can never become four in a row.
    """
    counts = [0.0, 0.0, 0.0]
    cells = board.cells
    for window in WINDOWS:
        own = opp = 0
        for r, c in window:
            v = cells[r][c]
            if v == 1:
                own += 1
            elif v == -1:
                opp += 1
        if own and not opp and own < 4:
            counts[own - 1] += 1
        elif opp and not own and opp < 4:
            counts[opp - 1] -= 1
    center = sum(cells[r][CENTER] for r in range(ROWS))
    raw = counts + [float(center)]
    return [raw[i] / SCALE[i] for i in range(4)]


class ValueModel:
    def __init__(self, weights=None, games_trained: int = 0):
        self.weights = [float(w) for w in weights] if weights else [0.0] * len(FEATURE_NAMES)
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError("expected %d weights, got %d" % (len(FEATURE_NAMES), len(self.weights)))
        self.games_trained = int(games_trained)

    def value(self, board: Board) -> float:
        """Value from player 1's point of view, in [-1, 1]."""
        if board.winner is not None:
            return float(board.winner)
        return math.tanh(sum(w * f for w, f in zip(self.weights, features(board))))

    def update(self, board: Board, target: float, lr: float) -> float:
        """One TD step: move V(board) towards `target`. Returns the error."""
        f = features(board)
        v = math.tanh(sum(w * x for w, x in zip(self.weights, f)))
        err = target - v
        grad = err * (1.0 - v * v)
        for i in range(4):
            self.weights[i] += lr * grad * f[i]
        return err

    def save(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"weights": self.weights, "games_trained": self.games_trained,
                       "features": FEATURE_NAMES}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ValueModel":
        with open(path) as fh:
            d = json.load(fh)
        if "weights" not in d:
            raise ValueError("%s is not a model file (no 'weights' key)" % path)
        if tuple(d.get("features", FEATURE_NAMES)) != FEATURE_NAMES:
            raise ValueError("%s was saved with different features %s" % (path, d["features"]))
        return cls(d["weights"], d.get("games_trained", 0))
