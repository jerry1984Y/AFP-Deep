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

