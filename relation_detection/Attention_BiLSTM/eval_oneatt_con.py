# coding: utf-8

import json
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Add, MaxPooling1D, \
    Concatenate, Dot, Flatten, RepeatVector, Multiply
from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K
import os
import tensorflow as tf
from configparser import ConfigParser
from keras.optimizers import Adam
from keras import losses
from sklearn.utils.class_weight import compute_class_weight
from relation_detection.Attention_BiLSTM.preprocess import readData
from relation_detection.Attention_BiLSTM.preprocess import readRelation

def ranking_loss(y_true, y_pred):
     return K.maximum(0.0, 0.1 + K.sum(y_pred*y_true,axis=-1))

def model_construct():
    # CONFIG
    config = ConfigParser()
    config.read('./config.ini')

    question_input = Input(shape=(config.getint('pre', 'question_maximum_length'), ), dtype='int32',name="question_input")
    relation_all_input = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input")
    relation_input = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input")

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
            trainable=False,name="sharedEmbd_r_w")
    relation_word_emd = sharedEmbd_r_w(relation_all_input)
    sharedEmbd_r = Embedding(relation_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_emd],
        input_length=config.getint('pre', 'relation_maximum_length'),
        trainable=True,name="sharedEmbd_r")
    relation_emd = sharedEmbd_r(relation_input)
    bilstem_layer = Bidirectional(LSTM(units=200, return_sequences=True, implementation=2),name="bilstem_layer")
    question_bilstm_1 = bilstem_layer(question_emd)
    # question_bilstm_2 = Bidirectional(LSTM(units=200, return_sequences=True, implementation=2),name="question_bilstm_2")(question_bilstm_1)
    relation_word_bilstm = bilstem_layer(relation_word_emd)
    relation_bilstm = bilstem_layer(relation_emd)
    # question_res = Add()([question_bilstm_1, question_bilstm_2])
    relation_con = Concatenate(axis=-2)([relation_word_bilstm, relation_bilstm])
    relation_res = MaxPooling1D(400, padding='same')(relation_con)
    relation_flatten = Flatten()(relation_res)

    fc_layer1 = Dense(400, use_bias=True, activation='tanh')
    fc_layer2 = Dense(1, use_bias=False, activation='softmax')
    rel_expand = RepeatVector(30)(relation_flatten)
    inputs = Concatenate()([question_bilstm_1, rel_expand])
    weights = fc_layer2(fc_layer1(inputs))
    question_att = MaxPooling1D(400, padding='same')(Multiply()([question_bilstm_1, weights]))

    # relation_maxpool = MaxPooling1D(400, padding='same')(relation_bilstm)
    # relation_word_maxpool = MaxPooling1D(400, padding='same')(relation_word_bilstm)
    # relation_res = Add()([relation_maxpool, relation_word_maxpool])
    result = Dot(axes=-1, normalize=True)([question_att, relation_flatten])
    model = Model(inputs=[question_input, relation_input, relation_all_input,], outputs=result)
    model.compile(optimizer=Adam(), loss=ranking_loss)
    return model

if __name__ == '__main__':
    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    neg_num = json.load(open('./neg_number.json', 'r'))
    relation_dict = json.load(open('./relation_dict.json', 'r'))
    new_relation_dict = {v: k for k, v in relation_dict.items()}

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)
    model = model_construct()
    model.load_weights('./runs/complex_oneatt/my_model_weights.h5')
    print(model.summary())

    question_feature = np.load('./test_question_feature.npy')

    relation_feature = np.load('./test_relation_feature.npy')
    relation_all_feature = np.load('./test_relation_all_feature.npy')

    print('positive data loaded...')
    simi_pos = model.predict([question_feature, relation_feature, relation_all_feature], batch_size=1024)

    print('positive similarity computed...')
    np.save('test_pre_pos.npy', simi_pos)

    relation_feature_neg = np.load('./test_relation_feature_neg.npy')
    relation_all_feature_neg = np.load('./test_relation_all_feature_neg.npy')

    print('negtive data loaded...')
    simi_neg = model.predict([question_feature, relation_feature_neg, relation_all_feature_neg], batch_size=1024)

    print('negtive similarity computed...')
    np.save('test_pre_neg.npy', simi_neg)

    acc = np.sum(simi_pos>simi_neg) / simi_pos.shape[0]
    print("relation pos>neg accurcy: " + str(acc))

    index = 0
    false_list = list()


    true_all = list()
    all_set = set()
    true_half = list()

    config = ConfigParser()
    config.read('./config.ini')
    data = readData(config.get('pre', 'test_filepath'))
    relation = readRelation(config.get('pre', 'relation_filepath'))

    for num, neg_index in neg_num:
        l = int(np.argmax(simi_neg[index: index + num])) # 最大负例下标
        max_neg = relation_feature_neg[index+l] # 选出的最优候选
        gold = relation_feature[index]



        if (max_neg == gold).all(): #判断最优候选是否与标准答案相同
            true_all.append(neg_index)
            #print(str(neg_index) + ",rel_right")
        else:
            false_list.append(neg_index)
            #print(str(neg_index) + ",rel_wrong")
        print(new_relation_dict[int(gold[0])] +","+ new_relation_dict[int(max_neg[0])]
             #+","+new_relation_dict[int(gold[1])] + "," + new_relation_dict[int(max_neg[1])]
        )



        index += num
        all_set.add(neg_index)




    print (data[0])
    print (relation[0][1])
    print (relation[1][1])
    print("len(true_all) == " + str(len(true_all)))
    print("len(true_half) == " + str(len(true_half)))
    print("len(false_list) == " + str(len(false_list)))
    print("len(all_set) == " + str(len(all_set)))

    print(true_all)
