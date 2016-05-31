
from __future__ import print_function
import numpy as np
import pickle
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributedDense, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb

max_features = 50
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32


print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

def train():

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    with open("save_weight_lstm.pickle", mode="wb") as f:
        pickle.dump(model.get_weights(),f)


def test():

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True))  # try using a GRU instead, for fun
    model.add(TimeDistributed(Dense(max_features)))
    model.add(Activation('softmax'))

    layeylstm_value = model.predict(X_test[:5])
    xx = np.argmax(layeylstm_value,axis=2)
    print (layeylstm_value.shape, xx.shape)

import seq2seq.models

if __name__ == "__main__":
    test()