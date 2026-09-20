"""Players. Every agent has choose(board, rng) -> column."""
from __future__ import annotations

import math
import random

from .game import Board
from .model import ValueModel


class Agent:
    name = "agent"

    def choose(self, board: Board, rng: random.Random) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def choose(self, board, rng):
        return rng.choice(board.legal_moves())


def winning_moves(board: Board, player: int):
    """Columns where `player` would win immediately."""
    out = []
    for col in board.legal_moves():
        b = board.copy()
        b.player = player
        if b.play(col).winner == player:
            out.append(col)
    return out


class GreedyAgent(Agent):
    """Wins if it can, blocks if it must, otherwise plays randomly."""
    name = "greedy"

    def choose(self, board, rng):
        wins = winning_moves(board, board.player)
        if wins:
            return rng.choice(wins)
        blocks = winning_moves(board, -board.player)
        if blocks:
            return rng.choice(blocks)
        return rng.choice(board.legal_moves())


class LearningAgent(Agent):
    """Picks the move whose resulting board the ValueModel likes most."""
    name = "learned"

    def __init__(self, model: ValueModel, epsilon: float = 0.0):
        self.model = model
        self.epsilon = epsilon

    def choose(self, board, rng):
        moves = board.legal_moves()
        if self.epsilon and rng.random() < self.epsilon:
            return rng.choice(moves)
        rng.shuffle(moves)  # random tie-break
        best, best_val = moves[0], -math.inf
        for col in moves:
            val = board.player * self.model.value(board.copy().play(col))
            if val > best_val:
                best, best_val = col, val
        return best


class _Node:
    __slots__ = ("board", "parent", "move", "children", "untried", "visits", "score")

    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.untried = board.legal_moves()
        self.visits = 0
        self.score = 0.0  # total reward for the player who moved INTO this node


class MCTSAgent(Agent):
    """Monte Carlo Tree Search with random playouts. Strong, but slow and does not learn."""
    name = "mcts"
    C = 1.4

    def __init__(self, simulations: int = 200):
        if simulations < 1:
            raise ValueError("simulations must be at least 1, got %r" % simulations)
        self.simulations = simulations

    def choose(self, board, rng):
        for player in (board.player, -board.player):  # take a win or block one outright
            forced = winning_moves(board, player)
            if forced:
                return rng.choice(forced)
        root = _Node(board.copy())
        for _ in range(self.simulations):
            node = root
            while not node.untried and node.children:  # select
                log_n = math.log(node.visits)
                node = max(node.children, key=lambda ch: ch.score / ch.visits
                           + self.C * math.sqrt(log_n / ch.visits))
            if node.untried:  # expand
                move = node.untried.pop(rng.randrange(len(node.untried)))
                node = _Node(node.board.copy().play(move), node, move)
                node.parent.children.append(node)
            b = node.board.copy()  # rollout
            while b.winner is None:
                b.play(rng.choice(b.legal_moves()))
            while node is not None:  # backpropagate
                node.visits += 1
                mover = -node.board.player
                node.score += 1.0 if b.winner == mover else (0.5 if b.winner == 0 else 0.0)
                node = node.parent
        return max(root.children, key=lambda ch: ch.visits).move


def make_agent(name: str, model: ValueModel = None, simulations: int = 200) -> Agent:
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name == "learned":
        return LearningAgent(model or ValueModel())
    if name == "mcts":
        return MCTSAgent(simulations)
    raise ValueError("unknown agent %r (use random, greedy, learned or mcts)" % name)
