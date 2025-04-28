import requests
import matplotlib.pyplot as plt
import copy
import random
from PIL import Image
import io

response = requests.get('https://sudoku-api.vercel.app/api/dosuku?query={newboard(limit:1){grids{value, solution}}}')
data = response.json()
sudoku = data['newboard']['grids'][0]['value']
solution = data['newboard']['grids'][0]['solution']
# print(sudoku)
# print(solution)


def break_row(grid, modified):
    while True:
        row = random.randint(0, 8)
        col1, col2 = random.sample(range(9), 2)
        if isinstance(grid[row][col1], int):
            if (row, col1) not in modified and (row, col2) not in modified:
                # print(f"{(row, col1)} and {(row, col2)} modified")
                grid[row][col2] = grid[row][col1] 
                modified.add((row, col1))
                modified.add((row, col2))
                return grid

def break_col(grid, modified):
    while True:
        col = random.randint(0, 8)
        row1, row2 = random.sample(range(9), 2)
        if isinstance(grid[row1][col], int):
           if (row1, col) not in modified and (row2, col) not in modified:
                # print(f"{(row1, col)} and {(row2, col)} modified")
                grid[row2][col] = grid[row1][col]
                modified.add((row1, col))
                modified.add((row2, col))
                return grid
           
def break_box(grid, modified):
    while True:
        box_row = random.choice([0, 3, 6])
        box_col = random.choice([0, 3, 6])
        cells = [(r, c) for r in range(box_row, box_row + 3)
                        for c in range(box_col, box_col + 3)]
        (r1, c1), (r2, c2) = random.sample(cells, 2)
        if isinstance(grid[r1][c1], int):
           if (r1, c1) not in modified and (r2, c2) not in modified:
                # print(f"{(r1, c1)} and {(r2, c2)} modified")
                grid[r2][c2] = grid[r1][c1]
                modified.add((r1, c1))
                modified.add((r2, c2))
                return grid

def generate_invalid_sudoku(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                grid[i][j] = " " 
    mistaken_grid = copy.deepcopy(grid)
    n_errors = random.randint(1,3)
    modified = set()
    for _ in range(n_errors):
        error_type = random.choice(["row", "col", "box"])
        # print(error_type)
        if error_type == "row":
            mistaken_grid = break_row(mistaken_grid, modified)
        elif error_type == "col":
            mistaken_grid = break_col(mistaken_grid, modified)
        else:
            mistaken_grid = break_box(mistaken_grid, modified)
        # print(modified)
    return mistaken_grid

def plot_image(grid): 
  mistaken_grid = generate_invalid_sudoku(grid)
  fig, ax = plt.subplots(figsize=(6, 6))

  # Draw the grid lines
  for i in range(10):
      lw = 2 if i % 3 == 0 else 1
      ax.plot([0, 9], [i, i], color='black', linewidth=lw)
      ax.plot([i, i], [0, 9], color='black', linewidth=lw)

  # Add the numbers
  for i in range(9):
      for j in range(9):
          num = mistaken_grid[i][j]
          ax.text(j + 0.5, 8.5 - i, str(num), va='center', ha='center', fontsize=12)

  # Set the limits and remove ticks
  ax.set_xlim(0, 9)
  ax.set_ylim(0, 9)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_aspect('equal')

  plt.tight_layout()
  # plt.show()
  fig.savefig('sudoku_plot.png', dpi=300)
  
  return Image.open('sudoku_plot.png').convert('RGB')


