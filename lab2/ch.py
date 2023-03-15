import argparse
import copy
import sys
import time
import math
from copy import deepcopy
import time

cache = {} # you can use this to implement state caching!
LARGE_NUM = 100000

class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    def __init__(self, board, turn):
        self.board = board
        self.turn = turn
        self.width = 8
        self.height = 8
        self.id = hash(str(board))

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")

    def get_winner(self):
        """
        first check if current state is terminal
        then check who wins based on checker rules
        return n if is not terminal, r for red win, b for black win.
        """
        # total count on all checkers
        r_count, b_count = 0, 0
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == "r" or self.board[i][j] == "R":
                    r_count += 1
                elif self.board[i][j] == "b" or self.board[i][j] == "B":
                    b_count += 1

        # both sides have remaining checkers and curr turn has no possible moves -> not yet finished
        # if curr turn has no available moves -> opponent wins
        # any with no checker means the opponent wins
        if self.get_possible_moves() is None:
            return get_next_turn(self.turn)
        # determine who wins
        elif r_count == 0:
            return 'b'
        elif b_count == 0:
            return 'r'
        else:
            return 'n'
            
    def utility(self, depth, turn):
        """
        Utility function to define how good this state is:
        for terminated state, winner is inf, loser is -inf
        for non-terminated state, use curr_score - opp_score
        where score = 4*king + 2*normal_checker_at_center + 1*other_normal
        """
        # find cache first
        if self.id in cache:
            if cache[self.id][1] == turn:
                if cache[self.id][0] == LARGE_NUM or cache[self.id][0] == -LARGE_NUM:
                    return cache[self.id][0] // (depth+1)
                else:
                    return cache[self.id][0]
            else:
                if cache[self.id][0] == LARGE_NUM or cache[self.id][0] == -LARGE_NUM:
                    return -cache[self.id][0] // (depth+1)
                else:
                    return -cache[self.id][0]

        score = 0
        # terminated states
        if (self.get_winner() == turn):
            cache[self.id] = (LARGE_NUM, turn)
            return LARGE_NUM // (depth+1)
        elif (self.get_winner() == get_next_turn(turn)):
            cache[self.id] = (-LARGE_NUM, turn)
            return -LARGE_NUM // (depth+1)

        # Evaluate the utility of current state
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == turn:
                    if (i <= 5 and i >= 2 and j <= 5 and j >= 2):
                        score += 2
                    else:
                        score += 1
                elif self.board[i][j] == turn.upper():
                    score += 4
                elif self.board[i][j] == get_opp_char(turn)[0]:
                    if (i <= 5 and i >= 2 and j <= 5 and j >= 2):
                        score -= 2
                    else:
                        score -= 1
                elif self.board[i][j] == get_opp_char(turn)[1]:
                    score -= 4
        cache[self.id] = (score, turn)
        return score

    def get_possible_moves(self):
        """
        A function of state class that outputs an array of all
        possible next-move states, return jump if there is jump available
        otherwise return simple moves
        """
        moves = []
        jumps = []

        # loop board
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j].lower() == self.turn:
                    if self.board[i][j].islower():
                        move_dirs = [(1, 1), (1, -1)] if self.turn == 'b' else [(-1, 1), (-1, -1)]
                        jump_dirs = [(2, 2), (2, -2)] if self.turn == 'b' else [(-2, 2), (-2, -2)]
                    else: # king
                        move_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                        jump_dirs = [(2, 2), (2, -2), (-2, 2), (-2, -2)]

                    # Check for possible jumps
                    for dir in jump_dirs:
                        new_i = i + dir[0]
                        new_j = j + dir[1]
                        if 0 <= new_i < self.height and 0 <= new_j < self.width and self.board[new_i][new_j] == '.':
                            # Check if there is an enemy checker to jump over
                            mid_i = (i + new_i) // 2
                            mid_j = (j + new_j) // 2
                            if self.board[mid_i][mid_j] in get_opp_char(self.turn):
                                new_state = State(deepcopy(self.board), deepcopy(self.turn))
                                new_state.board[i][j] = '.'
                                new_state.board[mid_i][mid_j] = '.'
                                
                                # if go to endline, change it to king
                                if ((new_i == 0 and self.turn == 'r') or (new_i == 7 and self.turn == 'b')):
                                    new_state.board[new_i][new_j] = self.board[i][j].upper()
                                else:
                                    new_state.board[new_i][new_j] = self.board[i][j]
                                
                                # recursion to find avaiable jumps, append only the final states
                                if jump_helper(jump_dirs, new_i, new_j, dir, new_state, jumps) is False:
                                    new_state.turn = get_next_turn(new_state.turn)
                                    temp = [new_state, 0]
                                    jumps.append(temp)

                    # skip normal moves if jumps exists
                    if (jumps):
                        continue
                    
                    # Check for possible moves
                    for dir in move_dirs:
                        new_i = i + dir[0]
                        new_j = j + dir[1]
                        if 0 <= new_i < self.height and 0 <= new_j < self.width and self.board[new_i][new_j] == '.':
                            new_state = State(deepcopy(self.board), get_next_turn(self.turn))
                            new_state.board[i][j] = '.'
                            # if go to endline, change it to king
                            if ((new_i == 0 and self.turn == 'r') or (new_i == 7 and self.turn == 'b')):
                                new_state.board[new_i][new_j] = self.board[i][j].upper()
                            else:
                                new_state.board[new_i][new_j] = self.board[i][j]
                            temp = [new_state, 0]
                            moves.append(temp)
        
        # If there are jumps, only return jumps
        if (jumps):
            return jumps
        else:
            return moves

# take the second element for sort
def take_second(elem):
    return elem[1]

def jump_helper(jump_dirs, i, j, dir, state, jumps):
    result = False
    for dir in jump_dirs:
        new_i = i + dir[0]
        new_j = j + dir[1]
        if 0 <= new_i < state.height and 0 <= new_j < state.width and state.board[new_i][new_j] == '.':
            # Check if there is an enemy checker to jump over
            mid_i = (i + new_i) // 2
            mid_j = (j + new_j) // 2
            if state.board[mid_i][mid_j] in get_opp_char(state.turn):
                new_state = State(deepcopy(state.board), deepcopy(state.turn))
                new_state.board[i][j] = '.'
                new_state.board[mid_i][mid_j] = '.'
                
                # if go to endline, change it to king
                if ((new_i == 0 and state.turn == 'r') or (new_i == 7 and state.turn == 'b')):
                    new_state.board[new_i][new_j] = state.board[i][j].upper()
                else:
                    new_state.board[new_i][new_j] = state.board[i][j]

                # recursion to find avaiable jumps, append only the final states
                result = True
                if jump_helper(jump_dirs, new_i, new_j, dir, new_state, jumps) is False:
                    new_state.turn = get_next_turn(new_state.turn)
                    temp = [new_state, 0]
                    jumps.append(temp)
    return result

def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']

def get_next_turn(curr_turn):
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'

def read_from_file(filename):
    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()
    return board

def sort_moves(array, reversed = False):
    """
    loop over all moves and calculate utility
    using sorted func to sort the moves to achieve node ordering
    depth is default 0
    """
    for move in array:
        move[1] = move[0].utility(0, move[0].turn)
    
    return sorted(array, key = take_second, reverse = reversed)

def find_winner_if_stalemate(state):
    # total count on all checkers
    r_count, b_count = 0, 0
    R_count, B_count = 0, 0
    for i in range(state.height):
        for j in range(state.width):
            if state.board[i][j] == "r":
                r_count += 1
            elif state.board[i][j] == "R":
                R_count += 1
            elif state.board[i][j] == "b":
                b_count += 1
            elif state.board[i][j] == "B":
                B_count += 1
    # compare result
    if (r_count + R_count > b_count + B_count):
        return 'r'
    elif (r_count + R_count == b_count + B_count and R_count < B_count):
        return 'r'
    elif (r_count + R_count == b_count + B_count and R_count == B_count):
        return 'd'
    else:
        return 'b'
    
def abp(state, depth, alpha, beta, max_depth, turn):
    # check cache
    # if (state.id+depth) in cache:
    #     return cache[state.id+depth]

    # if we have reached max_depth or if the game is over
    if depth == max_depth or state.get_winner() != 'n':
        return None, state.utility(depth, turn)

    # otherwise, expand the search tree
    best_move = None
    if turn == state.turn:
        v = -LARGE_NUM
        for temp in state.get_possible_moves():
            new_state = temp[0]
            _, val = abp(new_state, depth + 1, alpha, beta, max_depth, turn)
            if val > v:
                v = val
                best_move = new_state
            alpha = max(alpha, v)
            if alpha >= beta:
                break  # beta cut-off
        
    else: 
        v = LARGE_NUM
        for temp in state.get_possible_moves():
            new_state = temp[0]
            _, val = abp(new_state, depth + 1, alpha, beta, max_depth, turn)
            if val < v:
                v = val
                best_move = new_state
            beta = min(beta, v)
            if alpha >= beta:
                break  # alpha cut-off

    return best_move, v

def output_abp(state):
    steps = 0
    while True:
        state.display()
        if state.get_winner()=='n':
            if state.turn == 'r':
                # print("Red's turn")
                new_state, value = abp(state, 0, -LARGE_NUM, LARGE_NUM, 10, state.turn)
            else:
                # print("Black's turn")
                new_state, value = abp(state, 0, -LARGE_NUM, LARGE_NUM, 10, state.turn)
            state = new_state
        else:
            return
        # state.turn = get_next_turn(state.turn)
        # if too many steps taken, end the game
        steps += 1
        if (steps > 100):
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    turn = 'r'
    state = State(initial_board, turn)
    ctr = 0
    start = time.time()
    # output to file
    original_stdout = sys.stdout 	

    with open(args.outputfile, 'w') as f:
        # change output to file
        sys.stdout = f
        
        output_abp(state)

        # Reset the standard output
        sys.stdout = original_stdout

    end = time.time()
    print(end-start)