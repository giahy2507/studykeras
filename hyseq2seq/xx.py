from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import TimeDistributedDense, Activation

n_in = 684
n_out = 684
n_hidden = 512
n_samples = 2297
n_timesteps = 87

model = Sequential()
model.add(GRU(n_hidden, return_sequences=True, input_shape=(n_timesteps,n_in)))
model.add(TimeDistributedDense(n_out))
model.compile(loss='mse', optimizer='rmsprop')

X = np.random.random((n_samples, n_timesteps, n_in))
Y = np.random.random((n_samples, n_timesteps, n_out))

Xp = model.predict(X)
print Xp.shape
print Y.shape

model.fit(X, Y, nb_epoch=1)

hyhy = model.predict(X)
print hyhy.shape