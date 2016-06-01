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
batch_size = 10000

np.random.RandomState(4488)

X_train = np.random.randint(1,vocab_size-1, size=(batch_size,in_len -1))
y_train = np.random.randint(1,vocab_size-1, size=(batch_size,out_len -2))
X_train = sequence.pad_sequences(X_train,in_len,padding="post", truncating="post")
y_train = sequence.pad_sequences(y_train,out_len,padding="post", truncating="post")
y_train = to_mycategorical(y_train,vocab_size)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)

def nn_model(vocab_size, in_len, out_len, embsize, hidden_dim):

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

    return model

def train():

    model = nn_model(vocab_size, in_len, out_len, embsize, hidden_dim)

    earlystoping = EarlyStopping(patience=20)

    model.fit(X_train,y_train, nb_epoch=100, validation_split=0.2, callbacks=[earlystoping])

    with open("hyseq2seq.pickle", mode="wb") as f:
        pickle.dump(model.get_weights(), f)

def test(vocab_size, ):

    model = nn_model(vocab_size, in_len, out_len, embsize, hidden_dim)

    with open("hyseq2seq.pickle", mode="rb") as f:
        weights = pickle.load(f)

    model.set_weights(weights)

    hyhy = model.predict(X_train[-5:])

    print(hyhy.shape)

    label = np.argmax(hyhy,axis=2)
    target = np.argmax(y_train[-5:],axis=2)
    print label.shape

    print("Target:")
    print(target)

    print("Predict: ")
    print(label)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def preparedata():
    with open("data/movie_lines_cleaned_10k.txt", mode="r") as f:
        lines = f.readlines()

    his = [0]*500
    for line in lines:
        count = line.count(" ")
        his[count] += 1

    print("ttdt")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    X_data = pad_sequences(sequences,maxlen=25,padding="post",truncating="post")
    with open("data_tokenizer.pickle", mode="wb") as f:
        pickle.dump((X_data,tokenizer),f)

if __name__ == "__main__":

    with open("data_tokenizer.pickle", mode="rb") as f:
        X_data,tokenizer = pickle.load(f)

    vocab_size = 10000
    in_len = out_len = 25
    embsize = 100
    hidden_dim = 200


    X_train = X_data[:-1]
    y_train = X_data[1:]
    y_train = to_mycategorical(y_train,vocab_size)

    model = nn_model(vocab_size,in_len,out_len,embsize,hidden_dim)

    earlystoping = EarlyStopping(patience=20)

    model.fit(X_train,y_train,batch_size=32,nb_epoch=50,validation_split=0.2,callbacks=[earlystoping])

    with open("hyseq2seq.pickle", mode="wb") as f:
        pickle.dump(model.get_weights(), f)







