import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, RepeatVector, Activation, Dropout, TimeDistributed
from keras.preprocessing import sequence
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import pickle


def to_mycategorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    shape_tmp = y.shape
    if not nb_classes:
        nb_classes = np.max(y)+1

    result = np.zeros((y.shape[0],y.shape[1],nb_classes),dtype=np.int32)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            result[i,j,y[i,j]] = 1
    return result

vocab_size = 100
in_len = 20
out_len = 10
embsize = 25
hidden_dim = 50
batch_size = 50000

np.random.RandomState(4488)

X_train = np.random.randint(1,vocab_size-1, size=(batch_size,in_len -1))
y_train = np.random.randint(1,vocab_size-1, size=(batch_size,out_len -2))
X_train = sequence.pad_sequences(X_train,in_len,padding="post", truncating="post")
y_train = sequence.pad_sequences(y_train,out_len,padding="post", truncating="post")
y_train = to_mycategorical(y_train,vocab_size)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)




def train():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embsize, mask_zero=True))
    # shape ( None, in_len, embsize )


    # Encode state
    model.add(LSTM(hidden_dim))
    # shape (None, hidden_dim) --- get the last output
    model.add(Dropout(0.5))
    model.add(RepeatVector(out_len))
    # shape ( None, out_len, hidden_dim )


    # Decode state
    model.add(LSTM(hidden_dim, return_sequences=True))
    # shape (None, out_len, hidden_dim)

    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystoping = EarlyStopping(patience=20)

    model.fit(X_train,y_train, nb_epoch=100, validation_split=0.2, callbacks=[earlystoping])

    with open("hyseq2seq.pickle", mode="wb") as f:
        pickle.dump(model.get_weights(), f)

if __name__ == "__main__":
    train()





