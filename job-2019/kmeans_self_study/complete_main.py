# encoding: utf-8
from main import *
import numpy as np
def get_data_by_path(path):
     ## load data - > data_list
    data_list = get_data(path)
    return data_list
    
def get_input_embeding(data_list):
    ##get vocabulary
    word_list=get_words(data_list,1)
    #new_list = delete_stop_words(word_list)
    #make vocabulary list
    vocab= vocabulary(word_list)
    
    ##get input embedding
    VOCAB_SIZE = len(vocab)
    #词向量为300维
    EMBEDDING_SIZE= 300
    sentence_token,input_data=get_complete_data(data_list,vocab,20,VOCAB_SIZE,EMBEDDING_SIZE)
    return input_data,sentence_token

def evaluate_k_end(default_k):
    if default_k >100000:
        k =int(default_k/20)
        return k
        
    if default_k > 10000:
        k= int(default_k/8)
        return k
    
    if default_k > 3000:
        k = int(default_k/5)
        return k
    else:
        k = int(default_k/3)
        return k

    
def evaluate_k_start(default_k):
    if default_k >100000:
        k =20
        return k
        
    if default_k > 10000:
        k= 20
        return k
    
    if default_k > 3000:
        k =10
        return k
    else:
        k =3
        return k


        
    
def evaluate_k_between(default_k):
    if default_k >100000:
        k =20
        return k
        
    if default_k > 10000:
        k= 10
        return k
    
    if default_k > 3000:
        k =5
        return k
    else:
        k =1
        return k
    
def get_best_k(input_data):
    default_k=len(input_data)
    data_embeding,sentence_token=get_input_embeding(input_data)
    k_start=evaluate_k_start(default_k)
    k_end=evaluate_k_end(default_k)
    k_between=evaluate_k_between(default_k)
    ch_score_list,ch_i_list=calculate_calinski_harabaz_score(k_start,k_end,k_between,data_embeding)
    plt.figure()
    plt.plot(ch_i_list,ch_score_list)
    temp=0
    c=0
    for i in range(0,len(ch_i_list)):
        if c<ch_score_list[i]:
            c=ch_score_list[i]
            temp=i
    return ch_i_list[temp],data_embeding


def get_result_by_k(k,input_data,data_list):
    #k=163, 因此分为163类，获得相应的圆心和label
    label,cluster=kmeans(k,input_data)
    answer_index_list= list(input_data)
    #transform data
    label_list,cluster_list, default_data_list=data_transform(label,cluster,answer_index_list)
    #contribute data
    map_contribute_data_list=make_index_map(label_list,default_data_list)
    #获得所有表现特征向量的句子
    index=get_index_list(k,cluster_list,map_contribute_data_list)
    # k = len(index) 表示输出正确，每一个array是一个点
    get_execl(data_list,answer_index_list,index)
    print("finished")
    