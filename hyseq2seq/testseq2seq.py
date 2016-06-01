import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM, RepeatVector, Activation, Dropout, TimeDistributed
from keras.preprocessing import sequence
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import warnings
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

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    X_data = pad_sequences(sequences,maxlen=25,padding="post",truncating="post")
    with open("data_tokenizer.pickle", mode="wb") as f:
        pickle.dump((X_data,tokenizer),f)


class MyModelCheckPoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto'):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    with open(filepath, mode="wb") as f:
                        pickle.dump(self.model.get_weights(),f)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            with open(filepath, mode="wb") as f:
                pickle.dump(self.model.get_weights(),f)

if __name__ == "__main__":

    # preparedata()

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

    earlystoping = EarlyStopping(patience=10)
    modelcheckpoint = MyModelCheckPoint("hyseq2seq.model",verbose=1,save_best_only=True)

    model.fit(X_train,y_train,batch_size=32,nb_epoch=20,validation_split=0.2,callbacks=[earlystoping,modelcheckpoint])

    with open("hyseq2seq.pickle", mode="wb") as f:
        pickle.dump(model.get_weights(), f)







