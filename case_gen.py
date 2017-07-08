import pickle
import collections
import shogi
import time
import numpy as np

# making data from previous played games
print("opening logs")
s = time.time()
with open("wins.pick",'rb') as f:
    win = pickle.load(f)
with open("logs.pick",'rb') as f:
    log = pickle.load(f)
with open("boards.pick",'rb') as f:
    boards = pickle.load(f)
print("done in {} sec".format(time.time() - s))

total_case = len(log[1].keys())
train_case = 3*total_case/4
test_case = total_case - train_case
print("total:{}".format(total_case))
X, y = [], []
print("making np arrays")
s = time.time()
print(sum(win[1].values()))
print(sum(log[1].values()))
for k, v in log[1].iteritems():
    boards[k].ident()
    X.append(boards[k]._ident)
    y.append(float(win[1][k])/float(v))
X, y = np.array(X), np.array(y)
X.reshape(total_case, 13, 11)
y *= 10
print y, sum(y)/float(total_case)
print("{} elements created in {}".format(len(X), time.time() - s))
with open("X.pkl", 'wb') as f:
    X.dump(f)
with open("y.pkl", 'wb') as f:
    y.dump(f)