# Ultimate Tic-Tac-Toe AI Bot

This repository contains a Python AI bot for **Ultimate Tic-Tac-Toe**, a 9x9 variant of Tic-Tac-Toe. The AI uses **Minimax with alpha-beta pruning** to compute optimal moves and updates both the main and macro boards during gameplay.

---

## Features

- **Minimax Algorithm**: Computes the best move by simulating all possible future moves.
- **Alpha-Beta Pruning**: Speeds up Minimax by cutting off branches that will not affect the final decision.
- **Macro Board Management**: Tracks the outcome of each 3x3 subgrid (win, loss, or tie).
- **Randomized Move Selection**: Selects corners or center when multiple moves are equivalent.
- **Full Game Support**: Handles input/output and updates main and macro boards automatically.

---

## How the Algorithm Works

1. **Board Representation**:
   - **Main Board (9x9)**: Represents the full game grid where each move is placed.
   - **Macro Board (3x3)**: Represents each subgrid as won (`X` or `O`) or tied (`.`).

2. **Subgrid Selection**:
   - The AI determines which 3x3 subgrid it is allowed to play in based on the opponent’s last move and valid moves provided by the game engine.
   - If only one subgrid is valid, the AI plays there.
   - If multiple subgrids are possible, the AI chooses based on the macro board state.

3. **Move Evaluation**:
   - The AI runs **Minimax** to simulate every possible move in the selected subgrid.
   - **Minimax** recursively evaluates the board to a terminal state (win, loss, or tie) and assigns a score:
     - `+10` for AI win (minus depth for faster wins)
     - `-10` for opponent win (plus depth for slower losses)
     - `0` for tie
   - **Alpha-beta pruning** skips branches that cannot improve the outcome, speeding up the calculation.

4. **Choosing the Move**:
   - If the subgrid is mostly empty, the AI may choose a random corner or center to start strategically.
   - Otherwise, the AI selects the move with the highest Minimax score to maximize its chance of winning.

5. **Board Updates**:
   - After each move, the AI updates:
     - **Main board** with the new mark (`X` or `O`).
     - **Macro board** to reflect subgrid wins, losses, or ties.

---

## How to Run

1. Ensure Python 3 is installed.
2. Run the bot in an environment where it can read game input and output moves:

```bash
python ultimate_tictactoe_ai.py
