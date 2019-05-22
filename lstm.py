import pandas as pd
import numpy as np
import pickle
import gc
import scipy.sparse as sparse
import re
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Embedding, GlobalMaxPool1D, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, CuDNNLSTM,TimeDistributed, Dense, Concatenate, add
from keras.optimizers import SGD, Adam, Nadam, Adamax, RMSprop, Adagrad, Adadelta
from keras.models import Model
from keras.utils import to_categorical
from AttentionWithContext import AttentionWithContext
from SelfAttention import SeqSelfAttention


config = tf.ConfigProto()
jit_level = tf.OptimizerOptions.ON_1
config.graph_options.optimizer_options.global_jit_level = jit_level
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


with open("data/cookingClean/Cooking_Train.pkl", 'rb') as f:
    df_train = pickle.load(f)
with open("data/cookingClean/Cooking_Test.pkl", 'rb') as f:
    df_test = pickle.load(f)
with open("data/cookingClean/Cooking_Valid.pkl", 'rb') as f:
    df_valid = pickle.load(f)


labels = list(df_train)
labels.remove("text")
labels = ["text"] + labels


def fill_in_datasets(labs, df):    
    out = pd.DataFrame()
    for lab in labs :        
        if lab in list(df):
            out[lab] = df[lab]        
        else:
            out[lab] = (df.shape[0])*[0]
    return(out)

df_train = fill_in_datasets(labels, df_train)
df_test = fill_in_datasets(labels, df_test)
df_valid = fill_in_datasets(labels, df_valid)

labels = labels[1:]
n_labels = len(labels)


#from https://keras.io/examples/pretrained_word_embeddings/
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


#PARAMETERS
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100
output_size = n_labels
embedding_vector_length = 100
batch_size = 128
nb_epochs = 150


comments = df_train.text.tolist() + df_test.text.tolist() + df_valid.text.tolist()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(comments)
word_index = tokenizer.word_index


sequences_train = tokenizer.texts_to_sequences(df_train.text)
sequences_test = tokenizer.texts_to_sequences(df_test.text)
sequences_valid = tokenizer.texts_to_sequences(df_valid.text)

x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
x_valid = pad_sequences(sequences_valid, maxlen=MAX_SEQUENCE_LENGTH)

y_train = df_train.drop(df_train.columns[[0]], axis=1).values
y_test = df_test.drop(df_test.columns[[0]], axis=1).values
y_valid = df_test.drop(df_test.columns[[0]], axis=1).values

print('Shape of train tensor ', x_train.shape)
print('Shape of test tensor ', x_test.shape)
print('Shape of valid tensor ', x_valid.shape)


embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed


word_index = tokenizer.word_index

num_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, embedding_vector_length))

def bidir_gru(my_seq,n_units):
        return Bidirectional(CuDNNLSTM(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)

def create_model(embeddings, n_gru_layers=4, n_units=50, n_context=5, drop_rate = 0.4 ):
    
    sent_ints = Input(shape=(MAX_SEQUENCE_LENGTH,))
    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False,
                        )(sent_ints)
     
    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    x = sent_wv_dr
    if n_gru_layers > 1:
        sent_wa = []
        for k in range(n_gru_layers):
            x = bidir_gru(x, n_units)
            sent_wa.append(x)
        sent_wa = Concatenate()(sent_wa)
    else:
        sent_wa = bidir_gru(x, n_units)

    if n_context > 1:
        contxt_list = []
        for k in range(n_context):
            x, y  = AttentionWithContext(return_coefficients=True)(sent_wa)
            contxt_list.append(x)
        sent_att_vec = Concatenate()(contxt_list)
    else :
        sent_att_vec, y = AttentionWithContext(return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)
    pre_preds = Dense(units=2000,
                  activation='relu')(sent_att_vec)
    pre_preds = Dropout(drop_rate)(pre_preds)
    pre_pre_preds = Dense(units=200,
                  activation='relu')(pre_preds)
    pre_pre_preds = Dropout(drop_rate)(pre_pre_preds)

    preds = Dense(units=output_size,
                  activation='sigmoid')(pre_pre_preds)
    model = Model(sent_ints,preds)
    return model

model = create_model(embedding_matrix)
print(model.summary())

model.compile(loss='binary_crossentropy',
                  optimizer='adam')

my_callbacks = [EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose =1), ModelCheckpoint(filepath='params/cooking', 
                                       verbose=1, 
                                       save_best_only=True,
                                       save_weights_only=True)]

model.fit(x_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_data = (x_test, y_test),  callbacks = my_callbacks)

predicts = model.predict(x_valid)

with open('results/cooking_predict.txt', 'w') as file:
        file.write('id,pred\n')

        for idx,pred in enumerate(predicts):

            file.write(str(idx) + ',' + str(pred[0]) + ',')
            for predi in pred[1:-1] :
                 file.write(str(predi) + ',')
            file.write(str(pred[-1]) + '\n')


               