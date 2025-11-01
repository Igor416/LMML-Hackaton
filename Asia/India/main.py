import sys
import copy
import random

# Check if the entire board is full (no empty cells)
def board_full(grid):
    return all(cell != '' for row in grid for cell in row)

# Check if the given player has a winning line (row, column, or diagonal)
def has_tris(player, grid):
    # Check rows
    for row in grid:
        if row == [player]*3:
            return True
    # Check columns
    for col in range(3):
        if grid[0][col] == grid[1][col] == grid[2][col] == player:
            return True
    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] == player or grid[0][2] == grid[1][1] == grid[2][0] == player:
        return True
    return False

# Minimax algorithm to compute the best move
def minimax_play(is_max, grid, depth=0, alpha=-float('inf'), beta=float('inf')):
    # Base cases: check for win, loss or tie
    if has_tris('X', grid): return 10 - depth  # Maximizing player wins
    if has_tris('O', grid): return depth - 10  # Minimizing player wins
    if board_full(grid): return 0              # Tie

    moves = []
    # Loop through all empty cells to simulate possible moves
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == '':
                temp = copy.deepcopy(grid)
                temp[r][c] = 'X' if is_max else 'O'
                score = minimax_play(not is_max, temp, depth+1, alpha, beta)

                # Alpha-beta pruning
                if is_max: alpha = max(alpha, score)
                else: beta = min(beta, score)

                if alpha >= beta: break

                # Collect scores only at top level to choose the best move
                if depth == 0: moves.append([(r,c), score])
                else: moves.append(score)

    if not moves: return 0
    # Return the best move coordinates at top level, otherwise return the score
    if depth == 0:
        return list(max(moves, key=lambda x: x[1])[0] if is_max else min(moves, key=lambda x: x[1])[0])
    return max(moves) if is_max else min(moves)

# Place a mark ('X' or 'O') on the board
def place(player, pos, grid):
    r, c = pos
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == '':
        grid[r][c] = player

# Compute the top-left coordinates of the 3x3 subgrid containing the cell
def subgrid_start(r, c):
    return [r//3*3, c//3*3]

# Extract a 3x3 subgrid from the main board
def get_subgrid(grid, start_r, start_c):
    return [row[start_c:start_c+3] for row in grid[start_r:start_r+3]]

# Decide the next move for the current player
def choose_move(grid, offset=None):
    valid = get_empty_cells(grid)
    # If the board is empty, pick a random corner
    if len(valid) == 9: move = [random.choice([0,2]), random.choice([0,2])]
    # If only center is free in an almost full subgrid, choose it
    elif len(valid) == 8: move = [1,1] if [1,1] in valid else [random.choice([0,2]), random.choice([0,2])]
    # Otherwise, use Minimax to compute the optimal move
    else: move = minimax_play(True, copy.deepcopy(grid))
    # If the subgrid is part of a larger board, adjust the coordinates
    if offset:
        move[0] += offset[0]
        move[1] += offset[1]
    return move

# Get all empty cells on the given grid
def get_empty_cells(grid):
    return [[r,c] for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c]=='']

# Update the macro board (3x3) based on the latest move in the main board
def update_macro(player, pos, macro, grid):
    r, c = pos
    # Skip if coordinates are out of bounds
    if r < 0 or r >= 9 or c < 0 or c >= 9:
        return
    start_r, start_c = subgrid_start(r, c)
    sub = get_subgrid(grid, start_r, start_c)
    mark = None
    # Check if subgrid has a winner
    for p in ['X','O']:
        if has_tris(p, sub): mark = p
    # If subgrid is tied, mark as '.'
    if not mark and board_full(sub): mark = '.'
    # Place mark on macro board
    if mark:
        place(mark, [start_r//3, start_c//3], macro)

# Determine which subgrid to play in, based on valid moves
def compute_subgrid(macro, grid, valid_moves):
    xs = {m[0]//3 for m in valid_moves}
    ys = {m[1]//3 for m in valid_moves}
    # If only one subgrid is allowed, play there
    if len(xs)==1 and len(ys)==1:
        offset = subgrid_start(*valid_moves[0])
    else:
        # Otherwise, choose subgrid based on macro board
        offset = choose_move(macro)
        offset[0] *= 3
        offset[1] *= 3
    sub = get_subgrid(grid, *offset)
    return sub, offset

# --- MAIN GAME LOOP ---
print("BOT READY", file=sys.stderr)
macro_grid = [['' for _ in range(3)] for _ in range(3)]  # 3x3 macro board
main_grid = [['' for _ in range(9)] for _ in range(9)]   # 9x9 main board

while True:
    # Read opponent's move
    opp_r, opp_c = map(int, input().split())
    # Read number of valid moves and their coordinates
    n = int(input())
    valid_cells = [list(map(int, input().split())) for _ in range(n)]

    # Update main board with opponent's move
    place('O', [opp_r, opp_c], main_grid)
    update_macro('O', [opp_r, opp_c], macro_grid, main_grid)

    # Determine which subgrid to play in
    sub_grid, offset = compute_subgrid(macro_grid, main_grid, valid_cells)
    # Choose the optimal move in the subgrid
    my_move = choose_move(sub_grid, offset)

    # Update main and macro boards with our move
    place('X', my_move, main_grid)
    update_macro('X', my_move, macro_grid, main_grid)

    # Output the chosen move
    print(*my_move)
