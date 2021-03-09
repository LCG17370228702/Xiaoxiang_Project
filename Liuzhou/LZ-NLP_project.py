# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:54:04 2021

@author: LYSS
"""

import re
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


Path_train = "./data/train.csv"
Path_test = "./data/test.csv"
lab="###__###"

train = pd.read_csv(Path_train, sep = lab, header = None, encoding = 'utf-8')
test = pd.read_csv(Path_test, sep = lab, header = None, encoding = 'utf-8')

train.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']
test.columns = ['ID', 'Query_List']


train_Nodup = train.drop_duplicates()
train_Nodup = train.dropna()

test_Nodup = test.drop_duplicates()
test_Nodup = test.dropna()



def Del_web1(data):
    web = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
    data = re.sub(web, " ", data)
    return data.strip()

def Del_web2(data):
    web = re.compile('www.|WWW.|.com|.cn|.COM|.CN', re.S)
    data = re.sub(web, " ", data)
    return data.strip()



train_Nodup['Query_List'] = train_Nodup['Query_List'].apply(lambda x: Del_web1(x))
train_Nodup['Query_List'] = train_Nodup['Query_List'].apply(lambda x: Del_web2(x))

test_Nodup['Query_List'] = test_Nodup['Query_List'].apply(lambda x: Del_web1(x))
test_Nodup['Query_List'] = test_Nodup['Query_List'].apply(lambda x: Del_web2(x))


train_Nodup.to_csv("./data/processed_train.csv", header = None, encoding='utf-8')
test_Nodup.to_csv("./data/processed_test.csv", header = None, encoding='utf-8')