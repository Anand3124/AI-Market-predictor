import contextlib
import io
import os
import sys
import tempfile
import unittest

from connect4.__main__ import main


class CliTest(unittest.TestCase):
    def run_cli(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(argv)
        return out.getvalue()

    def test_rejects_non_positive_counts(self):
        for argv in (["arena", "random", "greedy", "--games", "0"],
                     ["train", "--games", "-1"],
                     ["train", "--eval-every", "0"],
                     ["play", "--opponent", "mcts", "--simulations", "0"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(argv)

    def test_train_then_arena(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.json")
            out = self.run_cli(["train", "--games", "20", "--eval-every", "10", "--eval-games", "2",
                                "--model", model, "--seed", "1"])
            self.assertIn("saved model to", out)
            self.assertTrue(os.path.exists(model))
            self.assertTrue(os.path.exists(os.path.join(d, "history.csv")))
            out = self.run_cli(["arena", "learned", "random", "--games", "4", "--model", model, "--seed", "1"])
            self.assertIn("over 4 games", out)

    def test_play_quits_cleanly_on_eof(self):
        stdin, sys.stdin = sys.stdin, io.StringIO("4\n")
        try:
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                main(["play", "--opponent", "greedy", "--seed", "0"])
        finally:
            sys.stdin = stdin
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
