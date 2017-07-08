from __future__ import print_function
import numpy as np
import pickle
import collections
import shogi
import time

# making data from previous played games
with open("X.pkl", 'rb') as f:
    X = pickle.load(f)
with open("y.pkl", 'rb') as f:
    y = pickle.load(f)
total_case = len(X)
train_case = 9*total_case/10
test_case = total_case - train_case
X_train, Y_train = X[0:train_case], y[0:train_case]
X_test, Y_test = X[train_case:], y[train_case:]
print("{} test cases, expected {}".format(len(X_test), test_case))

np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

batch_size = 10000
nb_epoch = 700

X_train = X_train.reshape(train_case, 13 * 11)
X_test = X_test.reshape(test_case, 13 * 11)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(512, input_dim=13*11))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',
              optimizer=Adam(lr=0.0000001),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('eval.hdf5')
print('Test score:', score[0])
print('Test accuracy:', score[1])