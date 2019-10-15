# encoding: utf-8
##vocabulary.py
from data_load import *
from collections import Counter
import jieba


#获得切完的词，和频率限制
def get_words(txt,min_frequency):
    vocab=[]
    c = Counter()
    for item in txt:
        seg_list = jieba.cut(item)
        for x in seg_list:
            if  x != '\r\n':
                c[x] += 1    
    for w, f in c.most_common():
        if f > 0:
            vocab.append(w)
    return vocab

#去除stopword，获得wordlist
def delete_stop_words(wordmax, stop_word_file='stopword_chinese.txt'):
    wordlist = []
    with open(stop_word_file,"r") as fp:
        words = fp.read()
        result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    stopword_set = set(new_words)
    for w in wordmax:
        if w not in stopword_set:
            wordlist.append(w)
    print("The trimed vocabulary is: {}".format(len(wordlist)))
    return wordlist


#构建属于自己的字典
def vocabulary(newlist):
    vocab = ["<pad>", "<unk>"]
    for w in newlist:
        vocab.append(w)
    print("The total size of our vocabulary is: {}".format(len(vocab)))
    return vocab

def get_vocabulary(data):
    list=get_words(data,1)
    #new_list = delete_stop_words(list)
    vocab_list=vocabulary(list)
    return vocab_list


if __name__ == '__main__':
    data = get_data("k_means_data.csv")
    vocab= get_vocabulary(data)