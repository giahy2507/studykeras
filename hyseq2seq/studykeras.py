__author__ = 'HyNguyen'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

import pickle

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train*1.0/255,dtype=np.float32).reshape(-1,28*28)
y_train = np.array(np_utils.to_categorical(y_train),dtype=np.int32)

X_test = np.array(X_test*1.0/255,dtype=np.float32).reshape(-1,28*28)
y_test = np.array(np_utils.to_categorical(y_test),dtype=np.int32)


def train():

    model = Sequential()
    model.add(Dense(output_dim=100, input_dim=28*28))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.fit(X_train,y_train)

    with open("save_weight.pickle", mode="wb") as f:
        pickle.dump(model.get_weights(),f)

def test():
    with open("save_weight.pickle", mode="rb") as f:
        weights = pickle.load(f)

    model = Sequential()
    model.add(Dense(output_dim=100, input_dim=28*28))
    model.add(Activation("relu"))
    model.set_weights(weights)

    layey1_value = model.predict(X_test[:5])
    y_pred = np_utils.categorical_probas_to_classes(y)
    Y = np_utils.categorical_probas_to_classes(y_test)
    print np_utils.accuracy(y_pred,Y)
    print y_pred.shape

if __name__ == "__main__":
    test()






