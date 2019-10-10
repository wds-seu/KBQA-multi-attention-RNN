# -*- coding:utf-8 -*-

import json
from type_classifier import stopword
import nltk
import sys

sys.path.append('..')
from nltk.stem import PorterStemmer

list_simple_id = {'1', '301', '2', '302'}
list_complex_id = {'3', '303', '5', '305', '6', '306', '9', '309', '7', '307', '8', '308', '15', '315', '16', '316'}
count_simple_id = {'101', '401', '102', '402'}
count_complex_id = {'103', '403', '105', '405', '106', '406', '111', '107', '108'}
ask_simple_id = {'151', '152'}


def is_count(question):
    if str(question).strip().lower().find("how many") != -1:
        return 1
    if str(question).strip().lower().find("count ") == 0:
        return 1
    if str(question).strip().lower().find("total number of") != -1:
        return 1
    return 0


def is_ask(question):
    ask = {'was', 'does', 'is', 'did', 'do', 'are', 'can', 'could'}
    word = str(question).strip().lower()
    word = word[0:word.find(' ')]
    if word in ask:
        return 1
    return 0


with open("../data/Lc_Quad-train.json", "r", encoding='utf-8') as file_data:
    data = json.loads(file_data.read())

    for data_i in data:
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        y = 0
        num_e = 0
        num_p = 0
        rest = ''

        question = str(data_i['question']).strip().rstrip("?").rstrip(".").strip().lower()
        sparql_id = str(data_i['sparql_id'])
        x2 = is_count(question)
        x3 = is_ask(question)
        x4 = len(question.split(' '))

        if str(data_i['entity2_uri']) == '':
            num_e = 1
            rest = stopword.wo_e_sentence(question, str(data_i['entity1_mention']))
        else:
            num_e = 2
            rest = stopword.wo_e_sentence(question, str(data_i['entity1_mention']))
            rest = stopword.wo_e_sentence(rest, str(data_i['entity2_mention']))
        x1 = num_e

        if sparql_id in list_simple_id:
            y = 0
        if sparql_id in list_complex_id:
            y = 0
        if sparql_id in count_simple_id:
            y = 1
        if sparql_id in count_complex_id:
            y = 1
        if sparql_id in ask_simple_id:
            y = 2

        tokens = stopword.seg_sentence(rest)
        tagged = nltk.pos_tag(tokens)

        num_V = 0
        num_N = 0
        num_J = 0
        num_tagged = 0
        num_1V_1N = 0


        for tagged_i in tagged:
            num_tagged += 1
            if tagged_i[1] in {'VBD', 'VBN', 'VBP', 'VBG', 'VBZ', 'VB'}:
                num_V += 1
            if tagged_i[1] in {'NN', 'NNS'}:
                num_N += 1
            if tagged_i[1] in {'JJ'}:
                num_J += 1


        # 根据x1 x2 x3 x4 进行规则分类

        print(str(x1)+"\t"+
              str(x2)+"\t"+
              str(x3)+"\t"+
              str(x4)+"\t"+
              str(num_tagged)+"\t"+
              str(num_V)+"\t"+
              str(num_N)+"\t"+
              str(num_J)+"\t"+
              str(y))