# encoding: utf-8
import warnings
warnings.filterwarnings("ignore")
from complete_main import *
def Input():
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input(' 请输入文件的正确路径> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            data_list= get_data(input_sentence)
            return data_list
        except Exception as e:
            print('except:', e)   
if __name__ == '__main__':
    data = Input()
     ##get k value
    k,input_embedding=get_best_k(data)
    #get result
    get_result_by_k(k,input_embedding,data)
    print("输入文件的目录为:answer.xlsx")
    
    