import shogi
import random


class MinimaxAgent:
    def __init__(self, eval):
        self.eval = eval
        self.result = None
        self.lookup = {}

    def minimax(self, side, board, depth, max_depth):
        mov_gen = shogi.Rule().gen_moves
        ev = self.eval
        if depth == max_depth:
            result = self.lookup.get(board.__hash__(), None)
            if result is None:
                self.lookup[board.__hash__()] = ev(board)
            return self.lookup[board.__hash__()]
        else:
            if side == 1:
                best = -1000
                moves = mov_gen(side, board)
                if len(moves) == 0:
                    return -1000
                for b in moves:
                    now = self.minimax(-side, b, depth+1, max_depth)
                    if now > best:
                        best = now
                        if depth == 0:
                            self.result = b
                return best
            else:
                best = 1000
                moves = mov_gen(side, board)
                if len(moves) == 0:
                    return 1000
                for b in moves:
                    now = self.minimax(-side, b, depth+1, max_depth)
                    if now < best:
                        best = now
                        if depth == 0:
                            self.result = b
                return best
            
    def make_move(self, side, board):
        self.result = None
        self.minimax(side, board, 0, 3)
        return self.result


class RandomAgent:
    import random
    def __init__(self):
        pass

    def make_move(self, side, board):
        randint = random.randint
        mov_gen = shogi.Rule().gen_moves
        l = mov_gen(side, board)
        return None if len(l) == 0 else l[randint(0, len(l)-1)]


import keras
model = keras.models.load_model('eval.hdf5')
def eval_nn(board):
    b = board.vectorize().reshape((1,13*11))
    p = model.predict
    return p(b,batch_size=1,verbose=0)[0][0]
m = MinimaxAgent(eval_nn)

def eval_mm(board):
    b = board.board
    y = 0
    result = 0
    while y < 4:
        x = 0
        while x < 3:
            if b[y][x] == -1:
                result -= 50
                if y == 2:
                    result -= 50
            elif b[y][x] == 1:
                result += 50
                if y == 0:
                    result += 50
            elif b[y][x] == -2:
                result -= 12/200.0
            elif b[y][x] == 2:
                result += 12/200.0
            elif b[y][x] == -3:
                result -= 10/200.0
            elif b[y][x] == 3:
                result += 10/200.0
            elif b[y][x] == -4:
                result -= 1/200.0
            elif b[y][x] == 4:
                result += 1/200.0
            elif b[y][x] == -5:
                result -= 11/200.0
            elif b[y][x] == 5:
                result += 11/200.0
            x += 1
        y += 1
    for p in board.captures:
        if p == 4 or p == -4:
            result += p/800.0
        elif p == 2 or p == -2:
            result += 3*p/200.0
        elif p == 3 or p == -3:
            result += 5*p/600.0
    return result
mm = MinimaxAgent(eval_mm)

class QlearnAgent:
    def __init__(self):
        self.model = keras.models.load_model("manmodel.h5")
        
    def make_move(self, side, board):
        mov_gen = shogi.Rule().gen_moves
        moves = mov_gen(side, board)
        if len(moves) == 0:
            return None
        q = []
        for b in moves:
            q.append(model.predict(b.vectorize().reshape((1,13*11))))
        if side == 1:
            return moves[q.index(max(q))]
        if side == -1:
            return moves[q.index(min(q))]

if __name__ == '__main__':
    
    b = shogi.Board([[-1,-2,-3],[0,0,0],[0, 0, 0],[1, 0, 0]],[])
    m.make_move(1,b)
    print(m.result)