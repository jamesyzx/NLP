# encoding: utf-8
import pandas as pd
def get_data(pathdir):
    data = pd.read_csv(pathdir)  
    data_list=list(data['sentence'])
    type(data_list)
    return data_list

if __name__ == '__main__':
    data = get_data("k_means_data.csv")