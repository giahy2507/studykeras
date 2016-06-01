import keras
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed,Embedding,LSTM, RepeatVector
from keras.preprocessing import sequence
import numpy as np

vocab_size = 100
in_len = 20
out_len = 10
embsize = 25
hidden_dim = 50
batch_size = 4488

np.random.RandomState(4488)

X_train = np.random.randint(1,vocab_size-1, size=(batch_size,in_len -1))
y_train = np.random.randint(1,vocab_size-1, size=(batch_size,out_len -2))
X_train = sequence.pad_sequences(X_train,in_len,padding="post", truncating="post")
y_train = sequence.pad_sequences(y_train,out_len,padding="post", truncating="post")

def train():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embsize,mask_zero=True))
    # shape ( None, in_len, embsize )

    model.add(LSTM(hidden_dim, dropout_U=0.2, dropout_W=0.2))
    # shape (None, hidden_dim) --- get the last output

    # model.add(RepeatVector(out_len))
    # shape ( None, out_len, hidden_dim )

    hyhy = model.predict(X_train[:5])
    print hyhy.shape

if __name__ == "__main__":
    train()





