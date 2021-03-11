# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr
import os

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)
    
def read_file(filename):
    """读取单个文件，文件为多类别"""
    contents = []
    labels = []
    with open_file(filename) as f :
        for line in f:
            try:
                raw = line.strip().split('\t')
                content = raw[1].split(' ')
                if content:
                    contents.append(content)
                    labels.append(raw[0])
            except:
                pass
    return contents, labels
    
def read_single_file(filename):
    """读取单个文件，文件为单一类别"""
    contents = []
    label = filename.split('/')[-1].split('.')[0]
    with open_file(filename) as f :
        for line in f:
            try:
                content = line.strip().split(' ')
                if content:
                    contents.append(content)
            except:
                pass
    return contents, label


def read_files(dirname):
    """读取文件夹"""
    contents = [] 
    labels = []
    files = [f for f in os.listdir(dirname) if f.endswith(".txt")]
    for filename in files :
        content,label = read_single_file(os.path.join(dirname,filename))
        contents.extend(content)
        labels.extend([label]*len(content))
    return contents, labels


def build_vocab(train_dir, vocab_file, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_files(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_file, mode='w').write('\n'.join(words) + '\n')
    


def read_vocab(vocab_file):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_file,mode='r') as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id



def encode_cate(content, words):
    """将id表示的内容转换为文字"""
    return[(words[x] if x in words else 40000) for x in content]

def encode_sentences(contents, words):
    """将id表示的内容转换为文字"""
    return [encode_cate(x,words) for x in contents]


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
