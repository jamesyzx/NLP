# encoding: utf-8
from flask import Flask
from flask import request
from flask import render_template
from flask import send_file
from requests import put, get
import time
from concurrent.futures import ThreadPoolExecutor
import json
from complete_main import *

# 构建一个线程池
executor = ThreadPoolExecutor(1)
# 创建一个flask对象,名字随意但一定要和下面的app对应
app = Flask(__name__) 

def a_b_sum(a,b):
    return a+b

# 访问根路径
@app.route('/')
def main():
    user = {'nickname': 'Miguel'} 
    return render_template("index.html", user = user,tag=a_b_sum)

@app.route('/run' ,methods={'POST',"GET"})
def getdata():
    inputdir=request.form.get("inputdir")
    outputdir=request.form.get("outputdir")
    print(inputdir)
    #/Users/james/NLP/job-2019/kmeans_self_study/data/k_means_data2.csv
    try:
        data=get_data_by_path(str(inputdir))
    except Exception as e:
        return "file can't found!!!!"
    executor.submit(do_update,data,inputdir)
    return  "process is running"
    
def do_update(data,inputdir):
    time.sleep(3)
    k,input_embedding=get_best_k(data)
    #get result
    get_result_by_k(k,input_embedding,data)    
    return "input:"+str(inputdir)+"outputdir:answer.xlsx"
    #return render_template("run_success.html",inputdir=inputdir,outputdir="输入文件的目录为:answer.xlsx")

# 执行py文件时，运行flask对象
if __name__ == '__main__':
    
    app.run(host='0.0.0.0',port = 80)