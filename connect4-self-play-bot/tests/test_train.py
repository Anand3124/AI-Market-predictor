import os
import random
import tempfile
import unittest

from connect4.agents import LearningAgent, RandomAgent
from connect4.arena import arena
from connect4.model import ValueModel
from connect4.train import train


class TrainTest(unittest.TestCase):
    def test_training_learns_and_saves(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            hist = os.path.join(d, "h.csv")
            model = train(300, path, eval_every=150, eval_games=20, seed=3,
                          history_path=hist, log=lambda *_: None)
            self.assertEqual(model.games_trained, 300)
            self.assertNotEqual(model.weights, [0.0] * 4)
            self.assertEqual(ValueModel.load(path).weights, model.weights)
            with open(hist) as fh:
                self.assertEqual(len(fh.read().strip().splitlines()), 3)
            # resuming continues from the saved count
            model = train(10, path, eval_every=10, eval_games=2, seed=4, log=lambda *_: None)
            self.assertEqual(model.games_trained, 310)

    def test_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            train(0, "unused.json", log=lambda *_: None)
        with self.assertRaises(ValueError):
            train(5, "unused.json", eval_every=0, log=lambda *_: None)
        with self.assertRaises(ValueError):
            train(5, "unused.json", eval_games=0, log=lambda *_: None)

    def test_trained_model_beats_random(self):
        with tempfile.TemporaryDirectory() as d:
            model = train(1500, os.path.join(d, "m.json"), eval_every=1500, eval_games=2,
                          seed=0, log=lambda *_: None)
        r = arena(LearningAgent(model), RandomAgent(), 100, random.Random(0))
        self.assertGreater(r["a"], 80, r)


if __name__ == "__main__":
    unittest.main()
