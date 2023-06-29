import os
from multiprocessing import Process
from utility import loadData


def threadprocesspssm(threadindex,dir_pssm,seq):
    for item in seq:
        label_content=item.split(',')
        if os.path.exists(os.path.join(dir_pssm, label_content[0])):
            continue
        file_object = open(os.path.join(dir_pssm, 'tmp' + str(threadindex) + '.fasta'), 'w')
        file_object.write(label_content[1])
        file_object.close()
        cmd = "./blastpgp -h 0.001 -j 3 -d ../../db/uniprot_sprot -i "
        cmd += os.path.join(dir_pssm,'tmp'+str(threadindex)+'.fasta')
        cmd += " -Q "
        cmd += os.path.join(dir_pssm, label_content[0])
        os.system(cmd)
    os.remove(os.path.join(dir_pssm,'tmp'+str(threadindex)+'.fasta'))

def pssm_progress(seq):
    dir_pssm="/home/dell/Documents/wujiashun/AFP-TLDeep/midData/PSSM"
    os.chdir('/home/dell/Documents/wujiashun/AFP-TLDeep/ncbi-blast-2.2.26/bin')
    keycount=len(seq)
    mthread_list = []
    threadcount = 30
    sectioncount = keycount // threadcount
    threadindex=0
    for i in range(threadcount):
        threadindex+=1
        if i == threadcount - 1:
            t = Process(target=threadprocesspssm,
                        args=(threadindex,dir_pssm,seq[sectioncount * i:keycount]))
            mthread_list.append(t)
        else:
            t = Process(target=threadprocesspssm,
                        args=(threadindex,dir_pssm,seq[sectioncount * i:sectioncount * (i + 1)]))
            mthread_list.append(t)
    for t in mthread_list:
        t.start()
    for t in mthread_list:
        t.join()
    print('PSSM finished')

if __name__ == '__main__':
    afplist=loadData('Dataset/afp.seq')
    nonafplist = loadData('Dataset/non-afp.seq')
    print(len(afplist))
    print(len(nonafplist))
    print('process afplist')
    pssm_progress(afplist)
    print('process non-afplist')
    pssm_progress(nonafplist)
    print('end')
