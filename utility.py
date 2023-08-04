import random

import pandas as pd
import numpy as np
import torch
def loadData(path):
    f = open(path)
    seqlist=f.readlines()
    return seqlist

def getgap():
    return 7


def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    dat = df[df.columns[1:]].values
    labels= df[df.columns[0]].values.astype(np.int32)
    features=dat.astype(np.float32)
    return features , labels

def read_csv_pssm(filename):
    df = pd.read_csv(filename, header=None)
    dat = df.values.astype(float).tolist()
    return torch.tensor(dat)

def read_file(filename):
    f = open(filename)
    seqlist = f.readlines()
    features=[]
    labels=[]
    for seq in seqlist:
        items=seq.split(',')

        features.append(torch.Tensor(list(map(float,items[1:]))).unsqueeze(1))
        labels.append(int(items[0]))
    f.close()
    return features,labels

def save_prob_label(probs,labels,filename):
    #data={'probs':probs,'labels':labels}
    probs = np.array(probs)
    labels = np.array(labels)
    data = np.hstack((probs.reshape(-1, 1), labels.reshape(-1, 1)))
    names = ['probs', 'labels']
    Pd_data = pd.DataFrame(columns=names, data=data)
    Pd_data.to_csv(filename)

def read_prob_label(filename):
    df = pd.read_csv(filename)
    probs = df[df.columns[1]].values.astype(np.float32)
    labels= df[df.columns[2]].values.astype(np.float32)
    return probs , labels

def create_list_train_test_balance():


    #f = open('Dataset/orderafp')
    f = open('Dataset/orderafp920')
    positive_all = f.readlines()
    f.close()
    random.shuffle(positive_all)

    #f = open('Dataset/ordernon_afp9493')
    f = open('Dataset/ordernon_afp3948')
    negative_all = f.readlines()
    f.close()
    random.shuffle(negative_all)
    lst_path_positive_train =positive_all[0:644] #300，644
    lst_path_negative_train =negative_all[0:644]#300，644 2763

    print("Positive train: ", len(lst_path_positive_train))
    print("Negative train: ", len(lst_path_negative_train))

    lst_positive_train_label = [1] * len(lst_path_positive_train)
    lst_negative_train_label = [0] * len(lst_path_negative_train)

    lst_path_train = lst_path_positive_train + lst_path_negative_train
    lst_label_train = lst_positive_train_label + lst_negative_train_label

    test_positive=positive_all[644:] #300，644
    test_negative=negative_all[644:]#300，644 2763

    test_positive_label= [1] * len(test_positive)
    test_negative_label = [0] * len(test_negative)

    test_path_data=test_positive+test_negative
    test_label=test_positive_label+test_negative_label

    return lst_path_train, lst_label_train,test_path_data,test_label

def create_list_train_test_imbalance():


    #f = open('Dataset/orderafp')
    f = open('Dataset/orderafp920')
    positive_all = f.readlines()
    f.close()
    random.shuffle(positive_all)

    #f = open('Dataset/ordernon_afp9493')
    f = open('Dataset/ordernon_afp3948')
    negative_all = f.readlines()
    f.close()
    random.shuffle(negative_all)
    lst_path_positive_train =positive_all[0:644] #300，644
    lst_path_negative_train =negative_all[0:2764]#300，644 2763

    print("Positive train: ", len(lst_path_positive_train))
    print("Negative train: ", len(lst_path_negative_train))

    lst_positive_train_label = [1] * len(lst_path_positive_train)
    lst_negative_train_label = [0] * len(lst_path_negative_train)

    lst_path_train = lst_path_positive_train + lst_path_negative_train
    lst_label_train = lst_positive_train_label + lst_negative_train_label

    test_positive=positive_all[644:] #300，644
    test_negative=negative_all[2763:]#300，644 2763

    test_positive_label= [1] * len(test_positive)
    test_negative_label = [0] * len(test_negative)

    test_path_data=test_positive+test_negative
    test_label=test_positive_label+test_negative_label

    return lst_path_train, lst_label_train,test_path_data,test_label

