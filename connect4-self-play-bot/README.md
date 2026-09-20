# Connect Four self-play bot

A Connect Four bot that gets better by playing against itself. Pure Python,
no dependencies, about 500 lines. Everything is in one small package so you
can read the whole thing in an evening.

## Quick start

```bash
git clone https://github.com/Anand3124/connect4-self-play-bot.git
cd connect4-self-play-bot

python -m connect4 play                       # play against the trained bot
python -m connect4 train --games 5000         # make it stronger
python -m connect4 arena learned greedy       # who wins over 100 games?
python -m unittest discover -s tests          # run the tests
```

Requires Python 3.8 or newer. Nothing to install.

## The four players

| name      | how it picks a move                                         | learns? |
|-----------|-------------------------------------------------------------|---------|
| `random`  | any legal column                                            | no      |
| `greedy`  | wins if it can, blocks if it must, otherwise random         | no      |
| `learned` | the column whose resulting board its value function rates best | **yes** |
| `mcts`    | takes wins and blocks, then Monte Carlo Tree Search with 200 random playouts | no      |

Ratings of the shipped model after 6000 training games, measured with
`python -m connect4 arena` over 200 games (they vary a few percent with `--seed`):

| match up            | result   |
|---------------------|----------|
| learned vs random   | 98% wins |
| learned vs greedy   | 70% wins |
| mcts vs learned     | 82% wins |

The bot climbed from about 15% to 70% against `greedy` while training; the full curve is in
`models/history.csv`, one row per 500 games. `mcts` is stronger but does not learn. Beating it is the next goal.

## How the learning works

`connect4/model.py` scores a board with four numbers, all counted as
"player 1 minus player -1":

1. **singles**: lines of 4 cells holding one piece and nothing of the opponent's
2. **pairs**: same, with two pieces
3. **triples**: same, with three pieces (one move from winning)
4. **center**: pieces in the middle column

The value of a board is `tanh(w1*singles + w2*pairs + w3*triples + w4*center)`,
a number between -1 (player -1 is winning) and +1 (player 1 is winning). The four
weights `w` are the whole model. They start at zero, which is why an untrained bot plays
randomly.

`connect4/train.py` makes the model play itself. After every move, the value of the
previous board is nudged towards the value of the new board, and at the end of a game
towards the real result. This is temporal-difference learning, TD(0), the same idea
TD-Gammon used to learn backgammon. Ten percent of training moves are random so the
bot keeps exploring.

Every 500 games it plays 100 games each against `random` and `greedy`, prints the
win rates and saves the model to `models/learned.json`. Training resumes from
that file, so run `train` again whenever you want more games.

## Layout

```
connect4/game.py     rules: board, legal moves, win and draw detection
connect4/model.py    features + value function + TD update, save/load as JSON
connect4/agents.py   random, greedy, learned and MCTS players
connect4/arena.py    play two agents for N games, alternating who starts
connect4/train.py    the self-play training loop and evaluation ladder
connect4/__main__.py the command line (train / arena / play)
tests/               unit tests for all of the above
models/              learned.json (weights) and history.csv (learning curve)
```

## Where to take it next

- **Better features.** Add "open threats" (a triple whose empty cell is playable
  right now) and watch `learned` overtake `greedy` by more.
- **Search plus learning.** Give `LearningAgent` a two-ply lookahead, or use the value
  function to guide MCTS instead of random playouts. That is the AlphaZero recipe in
  miniature.
- **Swap the linear model for a small neural network** with the same `value` /
  `update` / `save` / `load` interface and nothing else has to change.
- **Another game.** `Board` is the only game-specific class. Give Quoridor or
  Tic-tac-toe the same interface and every agent works on it.
