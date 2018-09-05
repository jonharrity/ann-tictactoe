"""Run this program with the bash command:
    python3 tictactoe.py
Or by replacing "python3" with a link to an executable version of 
python3 (ie python or /opt/python/python, etc)
Created by Jon Harrity"""



import random

EMPTY = ' '
#ADJ lists all three-in-a-row combinations that suffice to win
#these numbers were verified manually by running print_win_orientations()
ADJ = { (0,0): [[(1,0),(2,0)], [(0,1),(0,2)], [(1,1),(2,2)]],
        (1,0): [[(0,0),(2,0)], [(1,1),(1,2)]],
        (2,0): [[(0,0),(1,0)], [(2,1),(2,2)], [(1,1),(0,2)]],
        (0,1): [[(0,0),(0,2)], [(1,1),(2,1)]],
        (1,1): [[(0,1),(2,1)], [(1,0),(1,2)], [(0,0),(2,2)], [(0,2),(2,0)]],
        (2,1): [[(0,1),(1,1)], [(2,0),(2,2)]],
        (0,2): [[(0,0),(0,1)], [(1,2),(2,2)], [(1,1),(2,0)]],
        (1,2): [[(0,2),(2,2)], [(1,0),(1,1)]],
        (2,2): [[(0,2),(1,2)], [(2,0),(2,1)], [(0,0),(1,1)]] }
TILES = ADJ.keys()
MAX_DEPTH = 8

"""Board is a container class for game states. Methods includes 
the evaluation function for min/max, a deep clone method, method to print
out the current board, analysis of board (such as whether the game is over),
and method to place a move"""
class Board():
    def __init__(self):
        self.tiles = [[EMPTY for i in range(3)] for j in range(3)]
    def copy(self):
        new = Board()
        new.tiles = [[self.tiles[x][y] for y in range(3)] for x in range(3)]
        return new
    def get_empty_tiles(self):
        for i in TILES:
            if self[i] == EMPTY:
                yield i
    def update(self, move):# move (x, y, player)
        self.tiles[move[0]][move[1]] = move[2]
        key = (move[0],move[1])
        for line in ADJ[key]:
            if self[line[0]] == self[line[1]] == move[2]:
                self.winner = move[2]
    def __getitem__(self, key):
        return self.tiles[key[0]][key[1]]
    def __setitem__(self, key, item):
        self.tiles[key[0]][key[1]] = item
        return item


    """Eval method is the key part to the algorithm. 
    Scoring only takes place at terminal nodes, and provides 
    a score based upon whose turn it will be and how many 
    turns it will take to get to this game state."""
    def eval(self,x,y,depth,team, mark_max=15):
        key = (x,y)
        score = 0
        for a,b in ADJ[key]:#a and b are the adjacent squares
        #ie, a and b are on the same row/column/diagonal as (x,y)
            if self[a] == self[b] and (not self[a] == EMPTY):
                #team can win
                if team == 'x':
                    score += mark_max - depth
                elif team == 'o':
                    score += -(mark_max) + depth
        return score

    def is_empty(self, x, y):
        return self.tiles[x][y] == EMPTY
    def is_full(self):
        for n in self.get_empty_tiles():
            return False
        return True
    def is_done(self):
        try:
            if self.winner: return True
        except:
            if self.is_full():
                self.winner = None
                return True
            else:
                return False
    def print(self):
        for y in range(3):
            for x in range(3):
                print(self.tiles[x][y],end='')
                if not x == 2: print('|',end='')
            print('')
            if not y == 2:
                print('-----')
    def print_win_msg(self, human, ai):
        if self.winner == human:
            print('You won! Congratulations!!')
            self.print()
            print()
        elif self.winner == ai:
            print('Computer won! Better luck next time ;(')
            self.print()
            print()
        elif self.winner == None:
            print('Nobody won! You both are just too good...')
            print()

    """Return list of 18 integers, nine for x in order of TILES then nine for o,
each value in the range [0, 1]. If x has a mark at a tile, then the value for that input neuron will be 1, otherwise 0, and the same for o
First nine values are from supplied perspective, next nine are from other perspective"""
    def export_for_nn(self, perspective):
        if not (perspective == 'x' or perspective == 'o'):
            raise 'Unknown perspective: ' + perspective

        opponent_letter = {'x': 'o', 'o':'x'}[perspective]
        input_self = 0.9
        input_other = 0.1

        inputs = []
        my_value = {perspective: input_self}
        other_value = {'o': input_self}
        for tile in TILES:
            inputs.append(my_value.get(self[tile], input_other))
        for tile in TILES:
            inputs.append(other_value.get(self[tile], input_other))
        return inputs
            
    """Returns a float in range (0, 1) representing health of game board from supplied perspective. 0 = bad, 1 = good
    """
    def get_health(self, perspective):
        total = 0
        score = {perspective: 1}
        #add one to score for every nonblocked pair of perspective pieces
        #subtract one for the same of the other team
        for tile in TILES:
            for a,b in ADJ[tile]:
                if (not self[tile] == EMPTY) and ((self[tile] == self[a]) or (self[tile] == self[b])) and ((self[a] == EMPTY) or (self[b] == EMPTY)):
                    total += score.get(self[tile], -1)

        #scoring
        if total < -1:
            return 0.1
        elif total < 0:
            return 0.25
        elif total == 0:
            return 0.5
        elif total > 1:
            return 0.9
        elif total > 0:
            return 0.75


"""Used for debugging only: manually print out each
row/column/diagonal line relative to a specific square
for the purpose of verifying that the ADJ variable is 
hardcoded correctly."""
def print_win_orientations():
    for start in ADJ.keys():
        print('START: ')
        board = Board()
        board.update((start[0],start[1],'X'))
        board.print()
        print()
        for line in ADJ[start]:
            board = Board()
            board.update((start[0],start[1],'x'))
            board.update((line[0][0],line[0][1],'x'))
            board.update((line[1][0],line[1][1],'x'))
            board.print()
            print()
        if input('q for quit: ') == 'q':
            break

"""Ask the user where they want to move (wrapper for poll_user_move)
including safe checking for invalid input"""
def get_human_move(board):
    while 1:
        try:
            return poll_user_move(board)
        except Exception as ex:
            print('Some invalid input was supplied; x and y must be integers between 0 and 2 inclusive and should not be occupied yet on the board.')

def poll_user_move(board):
    x = int(input('Enter x value: '))
    while not (x >= 0 and x <= 2):
        x = int(input('Enter 0, 1, or 2 for x: '))
    y = int(input('Enter y value: '))
    while not (y >= 0 and y <= 2 and board.is_empty(x,y)):
        y = int(input('Enter 0, 1, or 2 for y that is not yet taken: '))
    return (x,y)

"""Used for testing results of the get_max function
(debugging only)"""
def test_board(board):
    for square in board.get_empty_tiles():
        cp = board.copy()
        cp.update((square[0],square[1],'o'))
        val = get_min(cp,square,2)
        print('for %s, result is %s'%(str(square),str(val)))
"""Used for testing results of the get_min function"""
def test_min(board):
    for square in board.get_empty_tiles():
        cp = board.copy()
        cp.update((square[0],square[1],'x'))
        val = get_max(cp,square,2)
        print('for %s, result is %s'%(str(square),str(val)))

"""The max function of the minimax algorithm. Implementation
uses a and b for alpha/beta pruning, calls the evaluation function
at terminal nodes, and picks the best moves selecting from calls
to the equivilant get_min function."""
team_switch = {'o':'x','x':'o'}
def get_max(board, tile, depth, a, b, team):
    if depth == MAX_DEPTH or board.is_done():
        return (tile, board.eval(tile[0],tile[1],depth,team))
    best = -1000
    best_square = None
    for square in board.get_empty_tiles():
        if a >= b:#here is the alpha/beta pruning part
            break
        cp = board.copy()
        cp.update((square[0],square[1],team))
        val = get_min(cp, square, depth+1, a, b, team_switch[team])[1]
        if val > a:
            a = val
        if val > best:
            best_square = square
            best = val
    return (best_square, best)

"""The min function of the minimax algorithm. Counterpart to the
get_max function."""
def get_min(board, tile, depth, a, b, team):
    if depth == MAX_DEPTH or board.is_done():
        return (tile, board.eval(tile[0],tile[1],depth,team))
    best = 1000
    best_square = None
    for square in board.get_empty_tiles():
        if a > b: 
            break
        cp = board.copy()
        cp.update((square[0],square[1],team))
        val = get_max(cp, square, depth+1, a, b, team_switch[team])[1]
        if val < b:
            b = val
        if val < best:
            best_square = square
            best = val
    return (best_square, best)


"""This is the "main" function of the program
which is run at the program start."""
def play_game():
    board = Board()
    turn_switch = {'x':'o', 'o':'x'}
    turn = ['o','x'][random.randint(0,1)]
    done = False
    while not done:
        if turn == 'x':
            print('Your turn. Current board:')
            board.print()
            board.update(get_human_move(board))
        else:
            print('Computer move; board:')
            board.print()
            move = get_max(board, None, 1, -1000, 1000)[0]
            board.update((move[0],move[1],'o'))
            print('Computer moved to (%s, %s)'%(str(move[0]),str(move[1])))
        done = board.is_done()
        turn = turn_switch[turn]
        print()
    board.print()
    print()
    board.print_win_msg()

if __name__ == '__main__':
    print('Welcome to tic tac toe! You are playing against a computer using the minimax algorithm with alpha/beta pruning. You are \'x\'; the computer is \'o\'.\n')
    play_game()



