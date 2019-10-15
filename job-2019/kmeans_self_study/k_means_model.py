# encoding: utf-8
#model 聚类算法
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
#  构造聚类器
def kmeans(n_clusters,input_data):
    estimator = KMeans(n_clusters)
    s =estimator.fit(input_data)
     #获得每个句子的簇
    return s.labels_,s.cluster_centers_
                     
                     
#获得比较k值的方法，用calinski_harabaz分数，值越大，表示效果越好                     
def calculate_calinski_harabaz_score(start,end,gap,input_data):
    score_list=[]
    i_list=[]
    for i in range(start,end,gap):
        clf = KMeans(n_clusters=i).fit(input_data)
        a=metrics.calinski_harabaz_score(input_data, clf.labels_)
        score_list.append(a)
        i_list.append(i)
    return score_list,i_list
        
    

    
def calculate_inertia_by_k(start,end,gap, input_data):    
    #迭代器，比较聚内平均距离
    score_list=[]
    i_list=[]
    for i in range(start,end,gap):
        clf = KMeans(n_clusters=i).fit(input_data) 
        score_list.append(clf.inertia_)
        i_list.append(i)
    return score_list,i_list