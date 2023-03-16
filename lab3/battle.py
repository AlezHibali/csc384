import argparse
import copy
import sys
import time

"""
0. (first time) S < > v ^ constraints -> fill in appropriately -> change everything to K
1. row, col constraints -> fill water and K -> if K filled, make four corners water
2. (only once per M) given M, go find up/down/left/right, if up/down is unavail or left/right has K, then fill left/right and change to K
3. if any K/water is added, go back to step 1; else add a K randomly and go step 1 (backtracking)
final: check for ship constraints, ok continue, not ok backtrack
"""

class Board:
    def __init__(self, grid, row_constraints, col_constraints, ship_constraints):
        self.row_constraints = row_constraints
        self.col_constraints = col_constraints
        self.ship_constraints = ship_constraints
        self.grid = grid
        self.size = len(grid)
        self.M_set = []
        
    def print_board(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.grid[i][j], end="")
            print()
        print()

    def count_ships(self):
        # count ships using dfs, return whether size satisfies
        ship_count = [0,0,0,0]
        visited = [[False] * self.size for _ in range(self.size)]

        def dfs(row, col, size):
            visited[row][col] = True
            for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row = row + i
                new_col = col + j
                if (is_valid(new_row, new_col, self.size) and visited[new_row][new_col] == False and self.grid[new_row][new_col] == "K"):
                    size = dfs(new_row, new_col, size + 1)
            return size

        for row in range(self.size):
            for col in range(self.size):
                if (visited[row][col] == False and self.grid[row][col] == "K"):
                    ship_count[dfs(row, col, 1)-1] += 1

        return (ship_count == self.ship_constraints)

    def M_operate(self):
        has_changed = False

        # no M exist -> nothing to be done
        if (len(self.M_set) == 0):
            return False

        # each time M needs to be modified:
        # add two K -> M change to K -> remove from M_set
        for temp in self.M_set:
            i = temp[0]
            j = temp[1]
            
            # FILL LEFT/RIGHT WITH K
            # M at top or bot row
            # OR its top/bot slot is filled water 
            # OR its left/right slot is filled K/M
            if (i == 0 or i+1 == self.size or self.grid[i+1][j] == "." or self.grid[i-1][j] == "."):
                self.fill_in_K(i,j-1)
                self.fill_in_K(i,j+1)
                self.grid[i][j] = "K"
                self.M_set.remove(temp)
                has_changed = True
            elif (self.grid[i][j-1] == "K" or self.grid[i][j-1] == "M"):
                self.fill_in_K(i,j+1)
                self.grid[i][j] = "K"
                self.M_set.remove(temp)
                has_changed = True
            elif (self.grid[i][j+1] == "K" or self.grid[i][j+1] == "M"):
                self.fill_in_K(i,j-1)
                self.grid[i][j] = "K"
                self.M_set.remove(temp)
                has_changed = True

            # FILL TOP/BOT WITH K
            # M at left or right col
            # OR its left/right slot is filled water 
            # OR its top/bot slot is filled K
            elif (j == 0 or j+1 == self.size or self.grid[i][j+1] == "." or self.grid[i][j-1] == "." \
                or self.grid[i-1][j] == "K" or self.grid[i+1][j] == "K"):
                self.fill_in_K(i-1,j)
                self.fill_in_K(i+1,j)
                self.grid[i][j] = "K"
                self.M_set.remove(temp)
                has_changed = True
            elif (self.grid[i-1][j] == "K" or self.grid[i-1][j] == "M"):
                self.fill_in_K(i+1,j)
                self.grid[i][j] = "K"
                self.M_set.remove(temp)
                has_changed = True
            elif (self.grid[i+1][j] == "K" or self.grid[i+1][j] == "M"):
                self.fill_in_K(i-1,j)
                self.grid[i][j] = "K"
                self.M_set.remove(temp)
                has_changed = True
        
        # true: something changes, need forward again
        # false: nothing changed, no need to forward
        if (has_changed):
            return True
        else:
            return False

    def init_prune(self):
        """
        0. (first time) S < > v ^ constraints -> fill in appropriately -> change everything to K
            ONLY EXECUTE ONCE
        """
        for i in range(self.size):
            for j in range(self.size):
                match self.grid[i][j]:
                    case "S":
                        self.fill_in_S(i,j)
                    case "<":
                        self.fill_in_left(i,j)
                    case ">":
                        self.fill_in_right(i,j)
                    case "^":
                        self.fill_in_top(i,j)
                    case "v":
                        self.fill_in_bot(i,j)
                    case "M":
                        self.fill_in_K(i,j)
                        self.grid[i][j] = "M"
                        self.M_set.append((i,j))

    def check_row_constraints(self):
        has_changed = False

        for i in range(self.size):
            # row_cons is -1 meaning this row is satisfied already
            if (self.row_constraints[i] == -1):
                continue;
            
            longest_ship = 0
            K_count = 0
            avail_count = 0

            # recording counts of K and available slots
            # also check the longest ship here
            # if longest exceeds 4, directly return false
            for j in range(self.size):
                if (self.grid[i][j] == "K" or self.grid[i][j] == "M"):
                    K_count += 1
                    longest_ship += 1
                    if (longest_ship > 4):
                        return -1
                elif (self.grid[i][j] == "0"):
                    avail_count += 1
                    longest_ship = 0
                elif (self.grid[i][j] == "."):
                    longest_ship = 0
                
            # check for constraints, and fill water/K
            if (self.row_constraints[i] < K_count):
                return -1
            elif (self.row_constraints[i] > K_count + avail_count):
                return -1
            elif (self.row_constraints[i] == K_count):
                self.row_constraints[i] = -1
                for j in range(self.size):
                    if (self.grid[i][j] == "0"):
                        self.grid[i][j] = "."
                        has_changed = True
            elif (self.row_constraints[i] == K_count + avail_count):
                self.row_constraints[i] = -1
                for j in range(self.size):
                    if (self.grid[i][j] == "0"):
                        self.fill_in_K(i,j)
                        has_changed = True

        # -1 for not meeting constraints, 0 for meeting but unchanged, 1 for changes        
        if (has_changed):
            return 1
        else:
            return 0
        
    def check_col_constraints(self):
        has_changed = False

        for j in range(self.size):
            # col_cons is -1 meaning this col is satisfied already
            if (self.col_constraints[j] == -1):
                continue;

            longest_ship = 0
            K_count = 0
            avail_count = 0

            # recording counts of K and available slots
            # also check the longest ship here
            # if longest exceeds 4, directly return false
            for i in range(self.size):
                if (self.grid[i][j] == "K" or self.grid[i][j] == "M"):
                    K_count += 1
                    longest_ship += 1
                    if (longest_ship > 4):
                        return -1
                elif (self.grid[i][j] == "0"):
                    avail_count += 1
                    longest_ship = 0
                elif (self.grid[i][j] == "."):
                    longest_ship = 0
                
            # check for constraints, and fill water/K
            if (self.col_constraints[j] < K_count):
                return -1
            elif (self.col_constraints[j] > K_count + avail_count):
                return -1
            elif (self.col_constraints[j] == K_count):
                self.col_constraints[j] = -1
                for i in range(self.size):
                    if (self.grid[i][j] == "0"):
                        self.grid[i][j] = "."
                        has_changed = True
            elif (self.col_constraints[j] == K_count + avail_count):
                self.col_constraints[j] = -1
                for i in range(self.size):
                    if (self.grid[i][j] == "0"):
                        self.fill_in_K(i,j)
                        has_changed = True

        # -1 for not meeting constraints, 0 for meeting but unchanged, 1 for changes        
        if (has_changed):
            return 1
        else:
            return 0
        
    def fill_in_K(self, i, j):
        self.grid[i][j] = "K"
        for x in [-1, 1]:
            for y in [-1, 1]:
                if (i+x == -1 or j+y == -1):
                    continue
                try:
                    self.grid[i+x][j+y] = "."
                except IndexError:
                    pass

    def fill_in_S(self, i, j):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (i+x == -1 or j+y == -1):
                    continue
                if (x == 0 and y == 0):
                    self.grid[i][j] = "K"
                else:
                    try:
                        self.grid[i+x][j+y] = "."
                    except IndexError:
                        pass
    
    def fill_in_left(self, i, j):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (i+x == -1 or j+y == -1):
                    continue
                if (x == 0 and y == 0):
                    self.grid[i][j] = "K"
                elif (x == 0 and y == 1):
                    temp = self.grid[i+x][j+y] 
                    self.fill_in_K(i+x,j+y)
                    if temp == ">" or temp == "M":
                        self.grid[i+x][j+y] = temp
                else:
                    try:
                        self.grid[i+x][j+y] = "."
                    except IndexError:
                        pass
    
    def fill_in_right(self, i, j):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (i+x == -1 or j+y == -1):
                    continue
                if (x == 0 and y == 0):
                    self.grid[i][j] = "K"
                elif (x == 0 and y == -1):
                    temp = self.grid[i+x][j+y] 
                    self.fill_in_K(i+x,j+y)
                    if temp == "<" or temp == "M":
                        self.grid[i+x][j+y] = temp
                else:
                    try:
                        self.grid[i+x][j+y] = "."
                    except IndexError:
                        pass

    def fill_in_top(self, i, j):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (i+x == -1 or j+y == -1):
                    continue
                if (x == 0 and y == 0):
                    self.grid[i][j] = "K"
                elif (x == 1 and y == 0):
                    temp = self.grid[i+x][j+y] 
                    self.fill_in_K(i+x,j+y)
                    if temp == "v" or temp == "M":
                        self.grid[i+x][j+y] = temp
                else:
                    try:
                        self.grid[i+x][j+y] = "."
                    except IndexError:
                        pass
    
    def fill_in_bot(self, i, j):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (i+x == -1 or j+y == -1):
                    continue
                if (x == 0 and y == 0):
                    self.grid[i][j] = "K"
                elif (x == -1 and y == 0):
                    temp = self.grid[i+x][j+y] 
                    self.fill_in_K(i+x,j+y)
                    if temp == "^" or temp == "M":
                        self.grid[i+x][j+y] = temp
                else:
                    try:
                        self.grid[i+x][j+y] = "."
                    except IndexError:
                        pass

    def find_empty_slot(self):
        res = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.grid[i][j] == "0"):
                    heuristics = self.row_constraints[i] + self.col_constraints[j]
                    res.append((i, j, heuristics))
        res.sort(key = eval)
        return res

    def forward(self):
        while (True):
            row = self.check_row_constraints()
            col = self.check_col_constraints()
            M_change = self.M_operate()
            if (row == -1 or col == -1):
                return False
            elif (row == 0 and col == 0 and M_change == False):
                return True

    def backtracking(self):
        # get all available slots, no available means solution found!
        # solution should pass ship constraints, otherwise fail
        empty = self.find_empty_slot()
        if (len(empty) == 0):
            if (self.count_ships()):
                return self.grid
            else:
                return None
        
        # backtrack
        for temp in empty:
            new_board = copy.deepcopy(self)
            new_board.fill_in_K(temp[0], temp[1])
            
            # after fill in K -> if forward bad, goto next empty;
            # if forward good, return its backtrack recursion
            # None stands for failure in this branch, thus goto next empty
            if (new_board.forward() == True):
                new_grid = new_board.backtracking()
                if (new_grid is None):
                    self.grid[temp[0]][temp[1]] = "."
                    continue
                else:
                    return new_grid
            else:
                self.grid[temp[0]][temp[1]] = "."
                if (self.forward() == True):
                    continue
                else:
                    return None

        # if nothing in "empty" satisfies, then go back
        return None

def eval(elem):
    return elem[2]

def is_valid(row, col, size):
    return 0 <= row < size and 0 <= col < size

def convert_to_output(grid):
    # output is init to be all water
    size = len(grid)
    visited = [[False] * size for _ in range(size)]
    output = [["."] * size for _ in range(size)]        

    # loop dir to find where and how long is the ship
    def find_dir_len(row, col):
        length = 1
        dir = 0 # 0 is S
        visited[row][col] = True
        for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row = row + i
            new_col = col + j
            dir = i + 2*j # 2right 1bot -2left -1top
            while (is_valid(new_row, new_col, size) and visited[new_row][new_col] == False and grid[new_row][new_col] == "K"):
                visited[new_row][new_col] = True
                length += 1
                new_row += i
                new_col += j

            if is_valid(new_row, new_col, size):
                visited[new_row][new_col] = True
                
            if length != 1:
                return length, dir
        return 1, 0

    for row in range(size):
        for col in range(size):
            if (visited[row][col] == False and grid[row][col] == "K"):
                # go all direction: if no direction, set S
                # if find next K, set <>^v and set next as M
                # if then no next, set it ><v^
                length, dir = find_dir_len(row, col)
                if (length == 1):
                    output[row][col] = "S"
                else:
                    match dir:
                        case 2: # right
                            output[row][col] = "<"
                            for i in range(length-2):
                                output[row][col+i+1] = "M"
                            output[row][col+length-1] = ">"
                        case 1: # bot
                            output[row][col] = "^"
                            for i in range(length-2):
                                output[row+i+1][col] = "M"
                            output[row+length-1][col] = "v"
                        case -2: # left
                            output[row][col] = ">"
                            for i in range(length-2):
                                output[row][col-i-1] = "M"
                            output[row][col-length+1] = "<"
                        case -1: # top
                            output[row][col] = "v"
                            for i in range(length-2):
                                output[row-i-1][col] = "M"
                            output[row-length+1][col] = "^"

    return output                        

def solve_battle_solitaire(board):
    # start = time.perf_counter()
    board.init_prune()
    board.forward()
    solution_grid = board.backtracking()
    solution_grid = convert_to_output(solution_grid)
    # end = time.perf_counter()
    # print(f"Finish in {end - start:0.4f} seconds")
    return solution_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', type=str, required=True, help='Path to the input file')
    parser.add_argument('--outputfile', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    with open(args.inputfile, "r") as f:
        row_constraints = list(map(int, f.readline().strip()))
        col_constraints = list(map(int, f.readline().strip()))
        ships = list(map(int, f.readline().strip()))
        grid = []
        for line in f:
            grid.append(list(line.strip()))

    board = Board(grid, row_constraints, col_constraints, ships)
    solution = solve_battle_solitaire(board)

    with open(args.outputfile, "w") as f:
        if solution is None:
            f.write("No solution found.")
        else:
            for i, row in enumerate(solution):
                f.write("".join(row))
                if i < len(solution) - 1:  # Check if this is not the last row
                    f.write("\n")

if __name__ == '__main__':
    main()