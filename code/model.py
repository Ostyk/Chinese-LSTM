import os
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import json
import tensorflow.keras as K

from typing import Tuple, List, Dict
from tensorflow.keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator
from sklearn.preprocessing import OneHotEncoder
from createdataset import *
from tensorflow.random import set_random_seed
set_random_seed(42)

####################
### loading data ###
####################
SUBSET = ['pku','msr']
PADDING_SIZE = 30
data, uni_word_to_idx, bi_word_to_idx = data_feed(SUBSET, PADDING_SIZE)

# saving critical info to be later used when used on test sets
predict_dict = dict()
predict_dict.update({"PADDING_SIZE": PADDING_SIZE})
predict_dict.update({"UNIGRAM_DICT": uni_word_to_idx})
predict_dict.update({"BIGRAM_DICT": bi_word_to_idx})

rel_path = '../resources/vocabs/'
if not os.path.exists(rel_path):
    os.mkdir(rel_path)
with open(rel_path + '_'.join(SUBSET) + "_" + str(PADDING_SIZE) + '.json', 'w') as f:
    json.dump(predict_dict, f)

#######################
### Hyperparameters ###
#######################
VOCAB_SIZE = {"unigrams": data['train']['info']['uni_VocabSize'],
              "bigrams": data['train']['info']['bi_VocabSize']}
EMBEDDING_SIZE = {"unigrams": 64,
                  "bigrams": 16}
HIDDEN_SIZE = 256
batch_size = 128
epochs = 20
LEARNING_RATE = 0.0015
INPUT_DROPOUT = 0.2
LSTM_DROPOUT = 0.45
RECURRENT_DROPOUT = 0.35

#######################
### Bulding model ###
#######################
K.backend.clear_session()

model_name = time.strftime('%Y-%m-%d_%H:%M:%S_%z')
def Model_A(vocab_size, embedding_size, hidden_size, PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT, LSTM_DROPOUT, RECURRENT_DROPOUT):
    print("Creating KERAS model")

    unigrams = K.layers.Input(shape=(None,))
    embedding_unigrams = K.layers.Embedding(vocab_size["unigrams"],
                                            embedding_size['unigrams'],
                                            mask_zero=True,
                                            name = 'embedding_unigrams')(unigrams)

    bigrams = K.layers.Input(shape=(None,))
    embedding_bigrams = K.layers.Embedding(vocab_size["bigrams"],
                                           embedding_size['bigrams'],
                                           mask_zero=True,
                                           name = 'embedding_bigrams')(bigrams)

    merged_vector = K.layers.concatenate([embedding_unigrams, embedding_bigrams], axis=-1, name = 'concatenated')

    BI_LSTM = (K.layers.Bidirectional(
               K.layers.LSTM(hidden_size, dropout = LSTM_DROPOUT,
                             recurrent_dropout = RECURRENT_DROPOUT,
                             return_sequences=True,
                             kernel_regularizer=K.regularizers.l2(0.01),
                             activity_regularizer=K.regularizers.l1(0.01)
                            ),name = 'Bi-directional_LSTM'))(merged_vector)

    predictions = K.layers.TimeDistributed(K.layers.Dense(4, activation='softmax'))(BI_LSTM)

    model = K.models.Model(inputs=[unigrams, bigrams], outputs=predictions)

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = K.optimizers.Adam(lr=LEARNING_RATE, clipnorm=1., clipvalue=0.5),
                  metrics = ['acc', K.metrics.Precision()])

    return model

model = Model_A(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE,
                PADDING_SIZE, LEARNING_RATE, INPUT_DROPOUT,
                LSTM_DROPOUT, RECURRENT_DROPOUT)
# Let's print a summary of the model
model.summary()

####################
### loggin model ###
####################
if not os.path.exists('../resources/report_images'):
    os.mkdir('../resources/report_images')

cbk = K.callbacks.TensorBoard('../resources/logging/keras_model_'+model_name)
print("\nStarting training...")
K.utils.plot_model(model, to_file='../resources/report_images/model.png')

early_stopping = K.callbacks.EarlyStopping(monitor='val_precision',
                              min_delta=0,
                              patience=3,
                              verbose=2, mode='auto')
csv_logger = K.callbacks.CSVLogger('../resources/logging/keras_model_'+model_name+'.log')
model_checkpoint = K.callbacks.ModelCheckpoint(filepath = '../resources/logging/keras_model_'+model_name+'.h5',
                                               monitor='val_precision',
                                               verbose=2,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='auto', period=1)
####################
### training     ###
####################

model.fit(data["train"]["X"], data["train"]["y"],
          epochs=epochs, batch_size=batch_size, shuffle=True,
          validation_data=(data["dev"]["X"], data["dev"]["y"]),
          callbacks=[cbk, csv_logger, model_checkpoint, early_stopping])
print("Training complete.\n")

rel_path = '../resources/models'
if not os.path.exists(rel_path):
    os.mkdir(rel_path)
weights = os.path.join(rel_path,'model_weights_'+model_name+'_'.join(SUBSET)+'.h5')
model_name_save = os.path.join(rel_path,'model_'+model_name+'_'.join(SUBSET)+'.h5')
model.save_weights(weights) #saving weights for further analysis
model.save(model_name_save)

####################
### Plotting     ###
####################


plot_training(model_name, True, PADDING_SIZE, epochs, '-'.join(SUBSET), size = 20)
