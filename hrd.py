from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys
import heapq

#====================================================================================

char_goal = '1'
char_single = '2'

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)

class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()


    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()
        

class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = depth + manhattan(self)
        self.depth = depth
        self.parent = parent
        self.id = hash(str(board.grid))  # The id for breaking ties.
    
    def __lt__(self, other):
        return self.f < other.f

def manhattan(state):
    for piece in state.board.pieces:
        if piece.is_goal:
            h = abs(piece.coord_x-1)+abs(piece.coord_y-3)
    return h

def is_goal_state(state):
    """
    Check if the current board is our goal state
    """
    for piece in state.board.pieces:
        if piece.is_goal and piece.coord_x == 1 and piece.coord_y == 3:
            return True
    return False

def generate_successors(state):
    """
    Generate Successors of current state
    Check for Available Moves in sequence of up, left, right, down
    For goal Caocao, in sequence of down, left, right, up
    """
    successors = []
    for piece in state.board.pieces:
        # 1x1 Pieces
        if piece.is_single:
            if piece.coord_y != 4 and state.board.grid[piece.coord_y+1][piece.coord_x] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_y += 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_y -= 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
            if piece.coord_x != 0 and state.board.grid[piece.coord_y][piece.coord_x-1] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_x -= 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_x += 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
            if piece.coord_x != 3 and state.board.grid[piece.coord_y][piece.coord_x+1] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_x += 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_x -= 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
            if piece.coord_y != 0 and state.board.grid[piece.coord_y-1][piece.coord_x] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_y -= 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_y += 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)

        # 2x2 Piece
        elif (piece.is_goal):
            if piece.coord_y != 0 and state.board.grid[piece.coord_y-1][piece.coord_x] == "." and state.board.grid[piece.coord_y-1][piece.coord_x+1] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_y -= 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_y += 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
            if piece.coord_x != 0 and state.board.grid[piece.coord_y][piece.coord_x-1] == "." and state.board.grid[piece.coord_y+1][piece.coord_x-1] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_x -= 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_x += 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
            if piece.coord_x != 2 and state.board.grid[piece.coord_y][piece.coord_x+2] == "." and state.board.grid[piece.coord_y+1][piece.coord_x+2] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_x += 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_x -= 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
            if piece.coord_y != 3 and state.board.grid[piece.coord_y+2][piece.coord_x] == "." and state.board.grid[piece.coord_y+2][piece.coord_x+1] == ".":
                # change curr state -> deep copy -> change back to normal -> continue loop
                piece.coord_y += 1
                new_board = Board(deepcopy(state.board.pieces))
                piece.coord_y -= 1
                new_state = State(new_board, 0, state.depth+1, state)
                successors.append(new_state)
        
        # 1x2 Pieces & 2x1 Pieces 
        else:
            # 2x1 Pieces 
            if (piece.orientation == 'v'):
                if piece.coord_y != 3 and state.board.grid[piece.coord_y+2][piece.coord_x] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_y += 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_y -= 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
                if piece.coord_x != 0 and state.board.grid[piece.coord_y][piece.coord_x-1] == "." and state.board.grid[piece.coord_y+1][piece.coord_x-1] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_x -= 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_x += 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
                if piece.coord_x != 3 and state.board.grid[piece.coord_y][piece.coord_x+1] == "." and state.board.grid[piece.coord_y+1][piece.coord_x+1] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_x += 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_x -= 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
                if piece.coord_y != 0 and state.board.grid[piece.coord_y-1][piece.coord_x] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_y -= 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_y += 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
            
            # 1x2 Pieces 
            else:
                if piece.coord_y != 4 and state.board.grid[piece.coord_y+1][piece.coord_x] == "." and state.board.grid[piece.coord_y+1][piece.coord_x+1] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_y += 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_y -= 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
                if piece.coord_x != 0 and state.board.grid[piece.coord_y][piece.coord_x-1] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_x -= 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_x += 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
                if piece.coord_x != 2 and state.board.grid[piece.coord_y][piece.coord_x+2] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_x += 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_x -= 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
                if piece.coord_y != 0 and state.board.grid[piece.coord_y-1][piece.coord_x] == "." and state.board.grid[piece.coord_y-1][piece.coord_x+1] == ".":
                    # change curr state -> deep copy -> change back to normal -> continue loop
                    piece.coord_y -= 1
                    new_board = Board(deepcopy(state.board.pieces))
                    piece.coord_y += 1
                    new_state = State(new_board, 0, state.depth+1, state)
                    successors.append(new_state)
    return successors

def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^': # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<': # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)
    
    return board

def DFS(initial_state, max_depth):
    stack = [initial_state]
    solution = []
    explored = set() # faster for checking existence
    while stack:
        curr_state = stack.pop()
        explored.add(curr_state.id)
        if curr_state.depth == max_depth: # pruning
            continue
        if is_goal_state(curr_state):
            while curr_state is not None:
                solution.append(curr_state)
                curr_state = curr_state.parent
            return reversed(solution)
        for neighbor_state in generate_successors(curr_state):
            if neighbor_state.id not in explored and neighbor_state.depth <= max_depth/2: # cutoff
                stack.append(neighbor_state)
    return None

def solution_output(board, out_file, algo):
    """
    Print Out Output and Write into Output File
    """
    init_state = State(board, 0, 0)
    if algo == "dfs":
        solution = DFS(init_state, 20000)
    elif algo == "astar":
        solution = A_star(init_state, 300)
    original_stdout = sys.stdout 	

    with open(out_file, 'w') as f:
        sys.stdout = f
        if solution is None:
            print("No solution found.")
            return

        for state in solution:
            state.board.display()
            print("")  # works as \n

        # Reset the standard output
        sys.stdout = original_stdout

def A_star(initial_state, max_depth):
    """
    A star implementation with multi-path pruning
    """
    heap = []
    heapq.heappush(heap, initial_state)
    solution = []
    explored = dict() # faster for checking existence
    
    while heap:
        curr_state = heapq.heappop(heap)
        # multi-path pruning
        if curr_state.id in explored and curr_state.f >= explored[curr_state.id]:
            continue
        explored[curr_state.id] = curr_state.f

        # pruning
        if curr_state.depth == max_depth: 
            continue

        # check if result
        if is_goal_state(curr_state):
            while curr_state is not None:
                solution.append(curr_state)
                curr_state = curr_state.parent
            return reversed(solution)

        # update frontier
        for neighbor_state in generate_successors(curr_state):
            heapq.heappush(heap,neighbor_state)

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)

    # algo choices
    solution_output(board, args.outputfile, args.algo)
