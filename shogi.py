import numpy as np
import copy
import random   
import time

LIO = 1
GIR = 2
ELE = 3
CHI = 4
HEN = 5


timer = {'copy':0, 'boardgen':0, 'boardeval':0}

def inX(x):
    return  0 <= x < 3

def inY(y):
    return 0 <= y < 4

def lio_move_gen(side, x, y):
    m = {((x + dx), (y + dy))\
         for dx in (-1, 0, 1) for dy in (1, -1) if inX(x + dx) and inY(y + dy)}
    m |= {(x + dx, y) for dx in (-1, 1) if inX(x + dx)}
    return m - {(x, y)}

def gir_move_gen(side, x, y):
    m = {(x + dx, y) for dx in (-1, 1) if inX(x + dx)}
    m |= {(x, y + dy) for dy in (-1, 1) if inY(y + dy)}
    return m - {(x, y)}

def ele_move_gen(side, x, y):
    return {(x + dx, y + dy) for dx in (-1, 1) for dy in (-1, 1)\
                            if inX(x+dx) and inY(y+dy)} - {(x, y)}

def chi_move_gen(side, x, y):
    return {(x, 0 if (y - side) < 0 else 3 if 3 < (y - side) else (y - side))}\
            - {(x, y)}

def hen_move_gen(side, x, y):
    m = {(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-side, 0)\
                        if inX(x+dx) and inY(y+dy)}
    m |= {(x, 0 if y + side < 0 else 3 if y + side > 3 else y + side)}
    return m - {(x, y)}
    

movegen = [0,
           lio_move_gen,
           gir_move_gen,
           ele_move_gen,
           chi_move_gen,
           hen_move_gen]


class Board:
    def __init__(self, board, captures):
        self.board = board
        self.captures = captures
        self._ident = None

    def ident(self):
        a = [[0 for i in range(11)] for j in range(4*3)]
        i = 0
        while i < 4:
            j = 0
            while j < 3:
                if self.board[i][j] != 0:
                    a[i*3+j][self.board[i][j]] = 1
                j += 1
            i += 1
        b = [0 for c in range(11)]
        for c in self.captures:
            b[c] = 1
        self._ident = a+[b]

    def move(self, m):
        s = time.time()
        dc = copy.deepcopy
        captures = dc(self.captures)
        board = dc(self.board)
        timer['copy'] += time.time() - s
        if len(m) == 3:
            k, i, j = m
            captures.remove(k)
            board[i][j] = k
            return Board(board, captures)
        else:
            i, j, di, dj = m
            if board[di][dj] != 0:
                if abs(board[di][dj]) == HEN:
                    captures.append(-board[di][dj]/HEN*CHI)
                else:
                    captures.append(-board[di][dj])
            if abs(board[i][j]) == CHI:
                if board[i][j] > 0 and di == 0:
                    board[di][dj] = HEN
                elif board[i][j] < 0 and di == 3:
                    board[di][dj] = -HEN
                else:
                    board[di][dj] = board[i][j]
            else:
                board[di][dj] = board[i][j]
            board[i][j] = 0
            return Board(board, captures)

    def vectorize(self):
        if self._ident is None:
            self.ident()
        return np.array(self._ident, dtype=np.float32)

    def __hash__(self):
        res = 0
        i = 1
        for l in self.board:
            for p in l:
                res += (p+6)*i
                i *= 1000
        for p in self.captures:
            res += (p+6)*i
            i *= 1000
        return res

    def __repr__(self):
        s = ''
        for l in self.board:
            s += str(l)+'\n'
        s += 'captures' + str(self.captures)
        return s

class Rule:
    def __init__(self):
        self.movegen = movegen

    def gen_moves(self, side, board):
        moves = []
        boards = []
        bapp = boards.append
        b = board.board
        captures = [i for i in board.captures if i*side > 0]
        i = 0
        while i < 4:
            j = 0
            while j < 3:
                if b[i][j] == 0:
                    moves += [(k, i, j) for k in captures]
                if b[i][j]*side > 0:
                    moves += [(i,j,y,x) for x, y in \
                            self.movegen[abs(b[i][j])](side,j,i)\
                            if b[y][x]*side <= 0]
                if b[i][j] == -side:
                    oli = i
                j += 1
            i += 1
        # Check
        s = time.time()
        for m in moves:
            nb = board.move(m)
            if not (self.isCheck(nb, side) or oli == 3*((1+side)/2)):
                bapp(nb)
        timer['boardeval'] += time.time() - s
        return boards 

    def isCheck(self, board, side):
        inx, iny = inX, inY
        b = board.board
        threats = [{LIO, ELE, HEN},
                   {LIO, GIR, CHI, HEN},
                   {LIO, ELE, HEN},
                   {LIO, GIR, HEN},
                   set(),
                   {LIO, GIR, HEN},
                   {LIO, ELE},
                   {LIO, GIR, HEN},
                   {LIO, ELE}]
        li, lj, f = 0, 0, 0
        while li < 4:
            lj = 0
            while lj < 3:
                if b[li][lj] == side:
                    f = 1
                    break
                lj += 1
            if f == 1:
                break
            li += 1
        i = 0
        """print '\n'
        print(board)
        print("lion@"),
        print(li, lj)"""
        while i < 9:
            di, dj = - 1 + i/3, i%3 - 1
            if inx(lj + dj) and iny(li + di):
                if b[li][lj] > 0:
                    if -b[li + di][lj + dj] in threats[i]:
                        #print(li+di, lj+dj, b[li+di][lj+dj])
                        return True
                else:
                    if b[li + di][lj + dj] in threats[-i-1]:
                        #print(li+di, lj+dj, b[li+di][lj+dj])
                        return True
            i += 1
        return False

if __name__ == '__main__':
    
    r = Rule()
    b = Board([[0,-1,0],[0,0,0],[0, 0, -4],[1, 0, 2]],[])
    for c in r.gen_moves(-1, b):
        print(c)