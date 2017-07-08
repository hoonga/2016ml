import collections
import pickle
from multiprocessing import Process, Lock, Value, Pipe
from multiprocessing.managers import SyncManager
import agents
import shogi
import time


class Referee:
    def __init__(self, p1, p2):
        self.player = {}
        self.player[1] = p1
        self.player[-1] = p2
        self.board = {}
        self.log = {1 : collections.defaultdict(int),
                    -1 : collections.defaultdict(int)}
        self.win = {1 : collections.defaultdict(int),
                    -1 : collections.defaultdict(int)}

    def match(self, rounds, pipe, pid, log_lev=1):
        if log_lev == 1:
            log = {1 : collections.defaultdict(int),
                   -1 : collections.defaultdict(int)}
            win = {1 : collections.defaultdict(int),
                   -1 : collections.defaultdict(int)}
            board = {}
        if log_lev == 2:
            win = collections.Counter()
        for R in range(rounds):
            if log_lev == 1:
                round = {1: collections.defaultdict(int),
                         -1: collections.defaultdict(int)}
            b = shogi.Board([[-2,-1,-3],[0,-4,0],[0,4,0],[3,1,2]],[])
            side = 1
            turn = 0
            while turn < 100:
                nb = self.player[side].make_move(side, b)
                if nb is None:
                    break
                if log_lev == 1:
                    board[nb.__hash__()] = nb
                    log[side][nb.__hash__()] += 1
                    round[side][nb.__hash__()]
                side = -side
                b = nb
                turn += 1
            if turn == 200:
                print "draw"
                continue
            winner = -side
            print(winner)
            if log_lev == 1:
                for b in round[winner].iterkeys():
                    win[winner][b] += 1
            if log_lev == 2:
                win[winner] += 1
        print("thread {} done".format(pid))
        if log_lev == 1:
            pipe.send((log, win, board))
        if log_lev == 2:
            print(win)
            #pipe.send(win)
            pass
        #pipe.close()

    def run(self, games, threads, log=1):
        print("making pipes and threads")
        p = [Pipe() for i in range(threads)]
        t = [Process(target=self.match,
                     args=(games/threads,
                           p[i][0],
                           i, log)) for i in range(threads)]
        print("running threads")
        for i in range(len(t)):
            t[i].start()
        for i in range(len(t)):
            if log == 1:
                l, w, b = p[i][1].recv()
            t[i].join()
            if log == 1:
                for side in (1, -1):
                    for k, v in l[side].iteritems():
                        self.log[side][k] += v
                    for k, v in w[side].iteritems():
                        self.win[side][k] += v
                for k, v in b.iteritems():
                    self.board[k] = v
        if log == 1:
            with open("wins.pick",'wb') as f:
                pickle.dump(self.win, f, pickle.HIGHEST_PROTOCOL)
            with open("logs.pick",'wb') as f:
                pickle.dump(self.log, f, pickle.HIGHEST_PROTOCOL)
            with open("boards.pick", 'wb') as f:
                pickle.dump(self.board, f, pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    r = Referee(agents.QlearnAgent(), agents.m)
    s = time.time()
    r.match(1, None, 0, 2)
    print(time.time()-s)