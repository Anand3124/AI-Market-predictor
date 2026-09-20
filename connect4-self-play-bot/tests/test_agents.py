import json
import os
import random
import tempfile
import unittest

from connect4.agents import GreedyAgent, LearningAgent, MCTSAgent, RandomAgent, make_agent
from connect4.arena import arena, play_game
from connect4.game import Board
from connect4.model import ValueModel, features


def board_from(cols):
    b = Board()
    for c in cols:
        b.play(c)
    return b


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1)

    def test_greedy_takes_win(self):
        b = board_from([0, 1, 0, 1, 0, 1])  # X to move, column 0 wins
        self.assertEqual(GreedyAgent().choose(b, self.rng), 0)

    def test_greedy_blocks(self):
        b = board_from([0, 6, 0, 6, 0])  # O to move, must block column 0
        self.assertEqual(GreedyAgent().choose(b, self.rng), 0)

    def test_mcts_takes_win(self):
        b = board_from([0, 1, 0, 1, 0, 1])
        self.assertEqual(MCTSAgent(50).choose(b, self.rng), 0)

    def test_mcts_blocks(self):
        b = board_from([0, 6, 0, 6, 0])
        self.assertEqual(MCTSAgent().choose(b, self.rng), 0)
        self.assertEqual(MCTSAgent(1).choose(b, self.rng), 0)

    def test_mcts_rejects_zero_simulations(self):
        with self.assertRaises(ValueError):
            MCTSAgent(0)

    def test_learned_takes_win_even_untrained(self):
        b = board_from([0, 1, 0, 1, 0, 1])
        self.assertEqual(LearningAgent(ValueModel()).choose(b, self.rng), 0)
        b = board_from([6, 0, 6, 0, 6, 0, 1])  # O to move, column 0 wins for O
        self.assertEqual(LearningAgent(ValueModel()).choose(b, self.rng), 0)

    def test_features_are_antisymmetric(self):
        b = board_from([3, 2, 3])
        flipped = Board()
        flipped.cells = [[-v for v in row] for row in b.cells]
        self.assertEqual(features(b), [-x for x in features(flipped)])
        self.assertEqual(features(Board()), [0.0, 0.0, 0.0, 0.0])

    def test_model_value_terminal_and_bounds(self):
        m = ValueModel([1, 1, 1, 1])
        self.assertEqual(m.value(board_from([0, 1, 0, 1, 0, 1, 0])), 1.0)
        self.assertTrue(-1 < m.value(board_from([3])) < 1)

    def test_model_validation(self):
        with self.assertRaises(ValueError):
            ValueModel([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as fh:
                json.dump({"games_trained": 3}, fh)
            with self.assertRaises(ValueError):
                ValueModel.load(path)
            ValueModel([1, 2, 3, 4], 7).save(path)
            m = ValueModel.load(path)
            self.assertEqual((m.weights, m.games_trained), ([1.0, 2.0, 3.0, 4.0], 7))

    def test_update_moves_value_towards_target(self):
        m = ValueModel()
        b = board_from([3])
        before = m.value(b)
        m.update(b, 1.0, 0.5)
        self.assertGreater(m.value(b), before)

    def test_arena_counts(self):
        r = arena(RandomAgent(), GreedyAgent(), 20, self.rng)
        self.assertEqual(r["a"] + r["b"] + r["draws"], 20)
        self.assertIn(play_game(RandomAgent(), RandomAgent(), self.rng), (1, -1, 0))

    def test_make_agent(self):
        for name in ("random", "greedy", "learned", "mcts"):
            self.assertEqual(make_agent(name).name, name)
        with self.assertRaises(ValueError):
            make_agent("nope")


if __name__ == "__main__":
    unittest.main()
