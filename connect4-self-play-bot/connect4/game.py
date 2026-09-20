"""Connect Four rules: a 6x7 board, two players (1 and -1), four in a row wins."""
from __future__ import annotations

ROWS, COLS, CONNECT = 6, 7, 4
CENTER = COLS // 2


def _all_windows():
    """Every group of 4 cells in a line. There are 69 of them on a 6x7 board."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * i, c + dc * i) for i in range(CONNECT)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    out.append(tuple(cells))
    return tuple(out)


WINDOWS = _all_windows()


class Board:
    """Row 0 is the bottom row. `player` is whoever moves next.

    `winner` is None while the game is running, 1 or -1 for a win, 0 for a draw.
    """

    def __init__(self):
        self.cells = [[0] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS
        self.player = 1
        self.winner = None
        self.moves = []

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.cells = [row[:] for row in self.cells]
        b.heights = self.heights[:]
        b.player = self.player
        b.winner = self.winner
        b.moves = self.moves[:]
        return b

    def legal_moves(self):
        if self.winner is not None:
            return []
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def play(self, col: int) -> "Board":
        if self.winner is not None:
            raise ValueError("game is over")
        if not 0 <= col < COLS or self.heights[col] >= ROWS:
            raise ValueError("illegal move: column %s" % col)
        row = self.heights[col]
        self.cells[row][col] = self.player
        self.heights[col] += 1
        self.moves.append(col)
        if self._wins_at(row, col):
            self.winner = self.player
        elif len(self.moves) == ROWS * COLS:
            self.winner = 0
        self.player = -self.player
        return self

    def _wins_at(self, row: int, col: int) -> bool:
        p = self.cells[row][col]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < ROWS and 0 <= c < COLS and self.cells[r][c] == p:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= CONNECT:
                return True
        return False

    def is_terminal(self) -> bool:
        return self.winner is not None

    def key(self) -> str:
        return "".join(str(c) for row in self.cells for c in row).replace("-1", "2")

    def render(self) -> str:
        sym = {1: "X", -1: "O", 0: "."}
        lines = [" ".join(sym[v] for v in row) for row in reversed(self.cells)]
        lines.append(" ".join(str(c + 1) for c in range(COLS)))
        return "\n".join(lines)
