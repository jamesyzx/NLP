# encoding: utf-8
from data_load import *
from vocabulary import *
from data import *
from k_means_model import *
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm

# 用来把所有的数据表按照每个簇，分到各自的簇
def make_index_map(test_label,test_list):
    map={}
    length = len(test_label)
    for i in tqdm(range(0,length)):
        key = str(test_label[i])
        if map.get(key) is not None:
            templist=map.get(key)
            templist.append(test_list[i])
            dicttemp={key:templist}
            map.update(dicttemp)
            
        else:
            newlist=[]
            newlist.append(test_list[i])
            newkey=str(test_label[i])
            dict={newkey:newlist}
            map.update(dict)
    return map

# 修改数据格式
def data_transform(label,cluster,answer_index_list):
    label_list=list(label)
    cluster_list =list(cluster)
    default_data_list =[]
    for i in answer_index_list:
        default_data_list.append(list(i))
    return label_list,cluster_list,default_data_list
    
#cos 相似度算法
def get_cossimi(x,y):
    myx=np.array(x) #将列表转化为数组，更好的数学理解是向量
    myy=np.array(y) #将列表转化为数组，更好的数学理解是向量
    cos1=np.sum(myx*myy) #cos(a,b)=a*b/(|a|+|b|)
    cos21=np.sqrt(sum(myx*myx))
    cos22=np.sqrt(sum(myy*myy))
    return cos1/(float(cos21*cos22))

# 获得最接近的值，从数组中获得相关cos最大值
# 输出最接近的点

def get_max_cos(cluster,input_data):
    temp = 0
    a=0
    for i in input_data:
        if a < get_cossimi(cluster,i):
            a=get_cossimi(cluster,i)
            temp= i
    return temp

## 把map的value_list 进行整理，然后输出
def get_index_list(k,cluster_list,map_contribute_data_list):
    index=[]
    for i in tqdm(range(0,k)):
        key = str(i)
        templist=map_contribute_data_list.get(key)
        point=get_max_cos(cluster_list[i],np.array(templist))
        index.append(point)
    return index

#获得最后的结果，通过index
def get_answer_index(data_list,answer_index_list,sentence):
    length = len(answer_index_list)
    for i in range (0,length):
        if (sentence==answer_index_list[i]).all():
            return i,data_list[i]
        
# 获得每个问句
def get_execl(data_list,answer_index_list,index):
    answer=[]
    for i in index:
        num,a = get_answer_index(data_list,answer_index_list,i)
        answer.append(a)
    data={'sentence':answer}
    df=DataFrame(data)
    df.to_excel('data/answer.xlsx')
    return answer
    
if __name__ == '__main__':
    ## load data - > data_list
    data_list = get_data("k_means_data1.csv")
    
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
    print("input size is : {}".format(input_data.shape))
    
    #获得句向量对应的index，用来输出句子的
    answer_index_list=list(input_data)
    # check the length of the answer list
    print("input_list length : {}".format(len(answer_index_list)))
    
    ##寻找最佳k值
    #计算calinski_harabaz的分数，获得拐点k值
    ch_score_list,ch_i_list=calculate_calinski_harabaz_score(3,10,1,input_data)
    plt.figure()
    plt.plot(ch_i_list,ch_score_list)
    
    #计算聚间距离
    i_score_list,i_list=calculate_inertia_by_k(3,10,1,input_data)
    plt.figure()
    plt.plot(i_list,i_score_list)
    
    #k=163, 因此分为163类，获得相应的圆心和label
    label,cluster=kmeans(7,input_data)
    
    #transform data
    label_list,cluster_list, default_data_list=data_transform(label,cluster,answer_index_list)
    #contribute data
    map_contribute_data_list=make_index_map(label_list,default_data_list)
    #获得所有表现特征向量的句子
    index=get_index_list(7,cluster_list,map_contribute_data_list)
    # k = len(index) 表示输出正确，每一个array是一个点
    print("index length : {}".format(len(index)))
    get_execl(data_list,answer_index_list)
    
    