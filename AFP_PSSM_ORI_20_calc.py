import math
import os
import numpy as np
import pandas as pd
import random
from utility import loadData


def pssm_normal_progress(seqlabel):
    file_pssm_dir = "midData/PSSM"
    dir_norm = "midData/PSSM_ORI_20"
    for eachfile in seqlabel:
        with open(os.path.join(file_pssm_dir, eachfile), 'r') as inputpssm:
            with open(os.path.join(dir_norm, eachfile), 'w') as outfile:
                count = 0
                po = ''
                for eachline in inputpssm:
                    count += 1
                    if count <= 3:
                        continue
                    if not len(eachline.strip()):
                        break
                    line=''
                    item=eachline.split()[2:22]
                    for everyitem in item:
                        if everyitem.strip() == '':
                            continue
                        line += everyitem.strip() + ','
                    line=line.rstrip(',')
                    po +=line+ '\n'
                outfile.write(po)
    print('PSSM normal finished')


if __name__ == '__main__':
    afplist=pd.read_csv('Dataset/AFP920.seq', header=None)
    nonafplist = pd.read_csv('Dataset/Non-AFP3948.seq', header=None)
    print(len(afplist))
    print(len(nonafplist))
    print('process afplist')
    afp=afplist[afplist.columns[0]].tolist()
    #pssm_normal_progress(afp)

    print('process non-afplist')
    nonafp=nonafplist[nonafplist.columns[0]].tolist()
    pssm_normal_progress(nonafp)
    print('end')