# coding: utf-8

import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Add, MaxPooling1D, \
    Concatenate, Dot, Flatten, Multiply, RepeatVector, Dropout
from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K
import os
import tensorflow as tf
from configparser import ConfigParser
from keras.optimizers import Adam
from keras import losses
from sklearn.utils.class_weight import compute_class_weight
import time
from keras.callbacks import ReduceLROnPlateau

def ranking_loss(y_true, y_pred):
    return K.maximum(0.0, 0.1 + K.sum(y_pred*y_true,axis=-1))


# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

# CONFIG
config = ConfigParser()
config.read('./config.ini')

# INPUT
question_input = Input(shape=(config.getint('pre', 'question_maximum_length'), ), dtype='int32',name="question_input")
relation_all_input = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input")
relation_input = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input")
relation_all_input_neg = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input_neg")
relation_input_neg = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input_neg")

# EMBEDDING
question_emd = np.load('./question_emd_matrix.npy')
relation_emd = np.load('./relation_emd_matrix.npy')
relation_all_emd = np.load('./relation_all_emd_matrix.npy')

question_emd = Embedding(question_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[question_emd],
        input_length=config.getint('pre', 'question_maximum_length'),
        trainable=False,name="question_emd")(question_input)

sharedEmbd_r_w = Embedding(relation_all_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_all_emd],
        input_length=config.getint('pre', 'relation_word_maximum_length'),
        trainable=True,name="sharedEmbd_r_w")

relation_word_emd = sharedEmbd_r_w(relation_all_input)

sharedEmbd_r = Embedding(relation_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_emd],
        input_length=config.getint('pre', 'relation_maximum_length'),
        trainable=True,name="sharedEmbd_r")

relation_emd = sharedEmbd_r(relation_input)

relation_word_emd_neg = sharedEmbd_r_w(relation_all_input_neg)
relation_emd_neg = sharedEmbd_r(relation_input_neg)

# Bi-LSTM
bilstem_layer = Bidirectional(LSTM(units=200, return_sequences=True, implementation=2),name="bilstem_layer")
question_bilstm_1 = bilstem_layer(question_emd)
question_dropout = Dropout(0.5)(question_bilstm_1)

relation_word_bilstm = Dropout(0.5)(bilstem_layer(relation_word_emd))
relation_bilstm = Dropout(0.5)(bilstem_layer(relation_emd))

relation_word_bilstm_neg = Dropout(0.5)(bilstem_layer(relation_word_emd_neg))
relation_bilstm_neg = Dropout(0.5)(bilstem_layer(relation_emd_neg))


relation_con = Concatenate(axis=-2)([relation_word_bilstm, relation_bilstm])
relation_res = MaxPooling1D(400, padding='same')(relation_con)
relation_flatten = Flatten()(relation_res)

relation_con_neg = Concatenate(axis=-2)([relation_word_bilstm_neg, relation_bilstm_neg])
relation_res_neg = MaxPooling1D(400, padding='same')(relation_con_neg)
relation_flatten_neg = Flatten()(relation_res_neg)

fc_layer1 = Dense(400, use_bias=True, activation='tanh')
fc_layer2 = Dense(1, use_bias=False, activation='softmax')

# Attention
rel_expand = RepeatVector(30)(relation_flatten)
inputs = Concatenate()([question_bilstm_1, rel_expand])
weights = fc_layer2(fc_layer1(inputs))
question_att = MaxPooling1D(400, padding='same')(Multiply()([question_dropout, weights]))

rel_expand_neg = RepeatVector(30)(relation_flatten_neg)
inputs_neg = Concatenate()([question_bilstm_1, rel_expand_neg])
weights_neg = fc_layer2(fc_layer1(inputs_neg))
question_att_neg = MaxPooling1D(400, padding='same')(Multiply()([question_dropout, weights_neg]))

# COSINE SIMILARITY
result = Dot(axes=-1, normalize=True)([question_att, relation_flatten])
result_neg = Dot(axes=-1, normalize=True)([question_att_neg, relation_flatten_neg])

out = Concatenate(axis=-1)([result, result_neg])

model = Model(inputs=[question_input, relation_input, relation_all_input,relation_input_neg, relation_all_input_neg ], outputs=out)
model.compile(optimizer=Adam(), loss=ranking_loss)

print(model.summary())
# quit()
train_question_features = np.load('./train_question_feature.npy')
train_relation_features = np.load('./train_relation_feature.npy')
train_relation_all_features = np.load('./train_relation_all_feature.npy')
train_relation_features_neg = np.load('./train_relation_feature_neg.npy')
train_relation_all_features_neg = np.load('./train_relation_all_feature_neg.npy')
train_labels = np.load('./train_label.npy')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

model.fit([train_question_features, train_relation_features, train_relation_all_features, train_relation_features_neg, train_relation_all_features_neg],
          train_labels,
          epochs=10,
          batch_size=1024,
          shuffle=True,
          callbacks=[reduce_lr])


timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print('\nModel writing to {}\n'.format(out_dir))

model.save_weights(os.path.join(out_dir, 'my_model_weights.h5'))

