# encoding: utf-8
from data_load import *
from vocabulary import *
import tqdm
import numpy as np
import tensorflow as tf
#构建映射
def get_index(vocab):
    # 单词到编码的映射，例如不错-> 45
    word_to_token = {word: token for token, word in enumerate(vocab)}
    # 编码到单词的映射，例如6 -> 股市
    token_to_word = {token: word for word, token in word_to_token.items()}
    return word_to_token

#拆分句子，形成词组组成的句子
def dividesentence(data):
    sentence = []
    seg_list = jieba.cut(data, cut_all=False)
    for item in seg_list:
        sentence.append(item)
    return sentence
  #把句子变成对应的token数字，且固定句子的长度，统一格式  
def convert_text_to_token(sentence, limit_size,word_to_token_map):
    """
    根据单词-编码映射表将单个句子转化为token
    
    @param sentence: 句子，str类型
    @param word_to_token_map: 单词到编码的映射
    @param limit_size: 句子最大长度。超过该长度的句子进行截断，不足的句子进行pad补全
    
    return: 句子转换为token后的列表
    """
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]
    
    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in dividesentence(sentence)]
    
    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]
    
    return tokens

#构建句子token
def get_sentence_token(data, SENTENCE_LIMIT_SIZE,word_to_token):
    sentence_token=[]
    for sentence in tqdm.tqdm(data):
        tokens = convert_text_to_token(sentence, SENTENCE_LIMIT_SIZE,word_to_token)
        sentence_token.append(tokens)
    return sentence_token

 #使用知乎预先训练的词向量
def loadword2vec(vocab):
    with open("/Users/james/Downloads/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5", 'r') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            # 当前单词
            curr_word = line[0]
            words.add(curr_word)
            # 当前词向量
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)
        print("have pretrained-vectors in vocab is: {}".format(len(set(vocab)&set(words))))
        print("do not have pretrained-vectors in vocab is : {}".format(len(set(vocab))-len(set(vocab)&set(words))))
        return word_to_vec


# 构建词向量矩阵
def word_matrix(VOCAB_SIZE, EMBEDDING_SIZE,word_to_token,word_to_vec):
     # 初始化词向量矩阵（这里命名为static是因为这个词向量矩阵用预训练好的填充，无需重新训练）
    static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])
    for word, token in tqdm.tqdm(word_to_token.items()):
    # 用glove词向量填充，如果没有对应的词向量，则用随机数填充
        word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
        static_embeddings[token, :] = word_vector

    # 重置PAD为0向量
    pad_id = word_to_token["<pad>"]
    static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)
    return static_embeddings


#构成数据集-句向量矩阵（20*300）,通过句向量求和来合成句向量，输出句向量numpy格式
def get_data_tokens(static_embeddings, sentence_token):
    embed = tf.nn.embedding_lookup(static_embeddings, sentence_token)
    # 相加词向量得到句子向量
    sum_embed = tf.reduce_sum(embed, axis=1)
    sess = tf.Session()
    with sess.as_default():
        result=sum_embed.eval()
    return result


def get_complete_data(data,vocab,sentence_length,VOCAB_SIZE, EMBEDDING_SIZE):
    word_to_token=get_index(vocab)
    sentence_token=get_sentence_token(data,sentence_length,word_to_token)
    word_to_vec=loadword2vec(vocab)
    static_embeddings=word_matrix(VOCAB_SIZE,300,word_to_token,word_to_vec)
    print("static embeddings shape is : {}".format(static_embeddings.shape))
    input=get_data_tokens(static_embeddings,sentence_token)
    return  sentence_token, input
if __name__ == '__main__':
    #data_list 是用list形式，可以后续用来查询输出的问句
    #data_list = get_data("k_means_data.csv")
    # 获得词汇表
    #vocab=get_vocabulary(data_list)
    VOCAB_SIZE = len(vocab)
    EMBEDDING_SIZE= 300
    #获得输入的句向量
    sentence_token,input_data=get_complete_data(data_list,vocab,20,VOCAB_SIZE,EMBEDDING_SIZE)
    print("input size is : {}".format(input_data.shape))