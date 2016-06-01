from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, RepeatVector, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import logging

import datetime
print("Started at: " + str(datetime.datetime.now()))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

xs = []

maxlen = 100
max_features=maxlen + 1
from numpy.random import shuffle

r = range(1, maxlen + 1, 1)

for i in range(1000):
    shuffle(r)
    new_x = r[::]
    xs.append(new_x)

def to_one_hot(id):
    zeros = [0] * max_features
    zeros[id] = 1
    return zeros

xs = np.asarray(xs)

ys = map(lambda x: map(to_one_hot, x), xs)
ys = np.asarray(ys)

print("XS Shape: ", xs.shape)
print("YS Shape: ", ys.shape)