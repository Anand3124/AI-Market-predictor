import unittest

from connect4.game import COLS, ROWS, WINDOWS, Board


class GameTest(unittest.TestCase):
    def test_window_count(self):
        self.assertEqual(len(WINDOWS), 69)

    def test_vertical_win(self):
        b = Board()
        for col in (0, 1, 0, 1, 0, 1, 0):
            b.play(col)
        self.assertEqual(b.winner, 1)
        self.assertEqual(b.legal_moves(), [])

    def test_horizontal_win_by_second_player(self):
        b = Board()
        for col in (0, 3, 0, 4, 0, 5, 1, 6):
            b.play(col)
        self.assertEqual(b.winner, -1)

    def test_diagonal_win(self):
        b = Board()
        for col in (0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3):
            b.play(col)
        self.assertEqual(b.winner, 1)

    def test_draw(self):
        b = Board()
        order = [0, 1, 2, 3, 4, 5, 6]
        # fill columns in a pattern that never makes four in a row
        pattern = [order, order, [1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 0], order, order]
        for row in pattern:
            for col in row:
                b.play(col)
        self.assertEqual(b.winner, 0)
        self.assertEqual(len(b.moves), ROWS * COLS)

    def test_illegal_moves(self):
        b = Board()
        for _ in range(ROWS):
            b.play(3)
        self.assertNotIn(3, b.legal_moves())
        with self.assertRaises(ValueError):
            b.play(3)
        with self.assertRaises(ValueError):
            b.play(COLS)

    def test_copy_is_independent(self):
        b = Board()
        c = b.copy().play(0)
        self.assertEqual(b.heights[0], 0)
        self.assertEqual(c.heights[0], 1)
        self.assertNotEqual(b.key(), c.key())


if __name__ == "__main__":
    unittest.main()
