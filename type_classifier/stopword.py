import jieba



# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  


# 对句子进行分词  
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())  
    stopwords = stopwordslist('stopword.txt')  # 这里加载停用词的路径
    outstr = []
    for word in sentence_seged:  
        if word not in stopwords:
            if word != ' ':
                outstr.append(word)
    return outstr

def wo_e_sentence(sentence , entity):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = entity.lower().strip(' ').split(' ')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != ' ':
                outstr = outstr+ " "+ word
    return outstr.strip()


