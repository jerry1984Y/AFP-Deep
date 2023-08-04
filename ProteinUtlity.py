import math
import os
import re
from collections import Counter

import propy
import propy.PseudoAAC
from numpy import genfromtxt
import numpy as np
import PssmFeatures

def writeAAC(seqfile,aacfold):
    print('startAAC process...')
    f = open(seqfile)
    lines=f.readlines()
    f.close()
    for line in lines:
        name,seq=line.split(',')
        name=name.strip()
        seq=seq.strip()
        seq=re.sub(r"[UZOBX]", "", seq)
        dict=propy.AAComposition.CalculateAAComposition(seq)
        lst = list(dict.values())
        file_object = open(os.path.join(aacfold, name), 'w')
        file_object.write(','.join("%.4f" % (aac/100) for aac in lst))
        file_object.close()

    print('AAC finished')

def writepseAAC(seqfile,pseaacfold):
    print('startPseAAC process...')
    f = open(seqfile)
    lines = f.readlines()
    f.close()
    for line in lines:
        name, seq = line.split(',')
        name = name.strip()
        seq = seq.strip()
        seq = re.sub(r"[UZOBX]", "", seq)
        pseaacdict=propy.PseudoAAC.GetAPseudoAAC(seq, 20,0.05)
        lst = list(pseaacdict.values())
        file_object = open(os.path.join(pseaacfold, name), 'w')
        file_object.write(','.join("%.4f" % (pseaac) for pseaac in lst))
        file_object.close()
    print('PseAAC finished')


def writeCS_MB_PSSM(seqfilename):
    seqfile=open(seqfilename)
    #negseq=open('Dataset/Non-AFP3948.seq')
    lines=seqfile.readlines()
    for line in lines:
        name,seq=line.split(',')
        name=name.strip()
        seq=seq.strip()
        pssmdata = genfromtxt("midData/NormPSSM/"+name, delimiter=',')
        seqlength=len(pssmdata)
        blocksize=seqlength//4
        features=''
        for i in range(0,4):
            protdict = {'A': 0,  'C': 0, 'D': 0,'E': 0,'F': 0,'G': 0,'H': 0,'I': 0,'K': 0,'L': 0,'M': 0,'N': 0,
                        'P': 0,'Q': 0, 'R': 0,  'S': 0, 'T': 0, 'V': 0, 'W': 0,'Y':0 }
            if i==3:
                eachblock=pssmdata[blocksize*i:,:]
                subseq=seq[blocksize*i:]
            else:
                eachblock=pssmdata[i*blocksize:(i+1)*blocksize,:]
                subseq = seq[i*blocksize:(i+1)*blocksize]
            #calc max
            maxindex=np.argmax(eachblock,axis=1)
            subseqlength=len(subseq)
            for letterindex in range(subseqlength):
                lettle=subseq[letterindex]
                if lettle in protdict:
                    protdict[lettle]+=maxindex[letterindex]
            plist=list(protdict.values())
            #print(plist)
            features+=','+','.join("%4f"% (ea/subseqlength) for ea in plist)
        file_object = open('midData/CSMBPSSM/'+name, 'w')
        features=features.lstrip(',')
        file_object.write(features)
        file_object.close()


def writepsepssm(pssmfilefold,psepssmfold):
    feature_type = "pse_pssm"
    print('startPsePssm process...')
    PssmFeatures.get_feature(pssmfilefold, feature_type, psepssmfold)

    print('PsePssm finished')

def writeDDE(seqfilename):
    """
    Function to generate dde encoding for protein sequences
    :param fastas:
    :param kw:
    :return:
    """
    AA =  'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    seqfile = open(seqfilename)
    lines = seqfile.readlines()
    for line in lines:
        name, seq = line.split(',')
        name = name.strip()
        sequence = seq.strip()
        sequence = re.sub(r"[UZOBX]", "", sequence)
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        features = ','.join("%4f" % ea for ea in tmpCode)
        file_object = open('midData/DDE/' + name, 'w')
        file_object.write(features)
        file_object.close()

def DPC(sequence):

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    tmpCode = [0] * 400
    for j in range(len(sequence) - 2 + 1):
        tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
            sequence[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    return tmpCode

def GAAC(sequence):
    """
    Function to generate gaac encoding for protein sequences
    :param fastas:
    :param kw:
    :return:
    """
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()
    code = []
    count = Counter(sequence)
    myDict = {}
    for key in groupKey:
        for aa in group[key]:
            myDict[key] = myDict.get(key, 0) + count[aa]

    for key in groupKey:
        code.append(myDict[key] / len(sequence))
    return code



