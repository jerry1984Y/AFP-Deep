import torch
import torch.nn as nn
import random
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import pandas as pd
import torch.nn.functional as F
import glob
import numpy as np

class TLDeepModel(nn.Module):
    def __init__(self):
        super(TLDeepModel, self).__init__()
        self.cnn1=nn.Conv1d(20,64,3,padding='same')
        self.relu=nn.ReLU()
        self.pool1=nn.AvgPool1d(3)

        self.cnn2=nn.Conv1d(64,128,3,padding='same')
        self.pool2=nn.AvgPool1d(3)

        self.cnn3 = nn.Conv1d(128, 128, 3,padding='same')
        self.pool3 = nn.AvgPool1d(3)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        self.fc1 = nn.Linear(128+1024, 256)
        self.fc_drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.fc_drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self,prot,pssm,data_length):

        pssm=pssm.permute(0,2,1)

        pssm=self.cnn1(pssm)
        pssm=self.relu(pssm)
        ddl=data_length#-2
        pssm=self.pool1(pssm)
        ddl=torch.floor((ddl-3)/3)+1

        pssm = self.cnn2(pssm)
        #ddl=ddl-2
        pssm = self.relu(pssm)
        ddl = torch.floor((ddl - 3) / 3) + 1
        pssm = self.pool2(pssm)

        pssm = self.cnn3(pssm)
        #ddl = ddl - 2
        pssm = self.relu(pssm)
        ddl = torch.floor((ddl - 3) / 3) + 1
        pssm = self.pool3(pssm)

        pssm = pssm.permute(0, 2, 1)


        pssm = torch.nn.utils.rnn.pack_padded_sequence(pssm, ddl.to('cpu'), batch_first=True)
        pssm, (h_n, h_c) = self.lstm(pssm)

        output = torch.cat((h_n[-1],h_n[-2]),dim=1)

        tsum = torch.sum(prot, dim=1, keepdim=False)  # batchsize,L
        leg = torch.unsqueeze(data_length, dim=1)
        prot = tsum / leg

        output=torch.cat((output,prot),dim=1)

        output = self.fc1(output)
        output=self.fc_drop1(output)
        output=self.fc2(output)
        output=self.fc_drop2(output)

        output = self.fc3(output)

        return output

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[1]), reverse=True)
    feature0 = []
    feature1 = []
    train_y = []

    for data in batch_traindata:
        feature0.append(data[0])
        feature1.append(data[1])
        train_y.append(data[2])
    data_length = [len(data) for data in feature1]
    feature0 = torch.nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    feature1 = torch.nn.utils.rnn.pad_sequence(feature1, batch_first=True, padding_value=0)
    return feature0,feature1,torch.tensor(train_y, dtype=torch.long),torch.tensor(data_length)

def create_list_train_test():
    # 'Dataset/orderafp'
    # 'Dataset/ordernon_afp'
    f = open('Dataset/orderafp920')
    #f = open('Dataset/orderafp')
    positive_all = f.readlines()
    f.close()
    random.shuffle(positive_all)

    f = open('Dataset/ordernon_afp3948')
    #f = open('Dataset/ordernon_afp9493')
    negative_all = f.readlines()
    f.close()
    random.shuffle(negative_all)
    lst_path_positive_train =positive_all[0:644] #300
    lst_path_negative_train =negative_all[0:2763] #300 2763

    print("Positive train: ", len(lst_path_positive_train))
    print("Negative train: ", len(lst_path_negative_train))

    lst_positive_train_label = [1] * len(lst_path_positive_train)
    lst_negative_train_label = [0] * len(lst_path_negative_train)

    lst_path_train = lst_path_positive_train + lst_path_negative_train
    lst_label_train = lst_positive_train_label + lst_negative_train_label

    test_positive=positive_all[644:] #300
    test_negative=negative_all[2763:] #300 2763

    test_positive_label= [1] * len(test_positive)
    test_negative_label = [0] * len(test_negative)

    test_path_data=test_positive+test_negative
    test_label=test_positive_label+test_negative_label

    return lst_path_train, lst_label_train,test_path_data,test_label
class BioinformaticsDataset(Dataset):
    # X: list cac file (full path)
    # Y: list label [0, 1]; 0: negative, 1: positive
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
        label = self.Y[index]

        df = pd.read_csv('midData/PSSM_ORI_20_ProtTran/' + self.X[index], header=None)
        dat = df.values.astype(float).tolist()
        dat = torch.tensor(dat)

        df2= pd.read_csv('midData/PSSM_ORI_20/' + self.X[index], header=None)
        dat2 = df2.values.astype(float).tolist()

        return dat,torch.tensor(dat2),label
    def __len__(self):
        return len(self.X)
def training_k_fold():
    #split_seed = 100
    #print("split_seed: ", split_seed)
    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_seed)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    # Split TRAIN_VAL dataset into 10-fold, 9 for training and 1 for cross validation
    fold = 0
    glo_arr_probs = []
    glo_arr_labels = []
    glo_arr_labels_pred = []
    for train_index, val_index in skf.split(lst_path_train_all, lst_label_train_all):
        # get train and crossvalidation data by splited indexes
        print("######### Fold : ", fold)
        # print("train_index: ", train_index)
        # print("val_index: ", val_index)
        X_train = [lst_path_train_all[i].strip() for i in train_index]
        X_val = [lst_path_train_all[i].strip() for i in val_index]
        Y_train = [lst_label_train_all[i] for i in train_index]
        Y_val = [lst_label_train_all[i] for i in val_index]

        # print("X_train indices: ", train_index)
        # print("X_val indices: ", val_index)
        print("Y_train len: ", len(Y_train), "Y_val len:", len(Y_val))

        train_set = BioinformaticsDataset(X_train, Y_train)
        val_set = BioinformaticsDataset(X_val, Y_val)

        #change the model for different framework cross-validation
        model = TLDeepModel()

        model=model.to(device)
        train_loader = DataLoaderX(dataset=train_set, batch_size=32,
                                  shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True,
                                  collate_fn=coll_paddding)
        val_loader = DataLoaderX(dataset=val_set, batch_size=32,
                                  shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True,
                                  collate_fn=coll_paddding)

        best_val_loss=1000
        loss_func =nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        epochs = 20

        for i in range(epochs):
            model.train()
            print('echo,',i)
            epoch_loss_train = 0.0
            nb_train = 0
            for data_x1, data_x2, data_y, data_length in train_loader:
                optimizer.zero_grad()
                y_pred = model(data_x1.to(device), data_x2.to(device), data_length.to(device))
                data_y = data_y.to(device)
                single_loss = loss_func(y_pred, data_y)
                single_loss.backward()
                optimizer.step()
                epoch_loss_train = epoch_loss_train + single_loss.item()
                nb_train += 1
            epoch_loss_avg=epoch_loss_train/nb_train
            print("epoch_loss_avg: ", epoch_loss_avg)
        #val test
        model.eval()
        # epoch_loss=0
        # nb_test=0
        with torch.no_grad():
            for data_x1, data_x2, data_y, data_length in val_loader:
                y_pred = model(data_x1.to(device), data_x2.to(device), data_length.to(device))
                y_pred = torch.nn.functional.softmax(y_pred, dim=1)
                glo_arr_probs.extend(y_pred[:, 1].to('cpu'))
                y_pred = torch.argmax(y_pred, dim=1).to('cpu')
                glo_arr_labels.extend(data_y)
                glo_arr_labels_pred.extend(y_pred)
        fold += 1
    #base on threshold from 0.3-0.8 step 0.01,to calc max mcc
    print('begin calc')
    auc = metrics.roc_auc_score(glo_arr_labels, glo_arr_probs)
    maxmcc=0
    maxsn=0
    maxsp=0
    maxf1=0
    maxyouden=0
    maxacc=0
    balanceacc=0
    thr=0
    for i in range(30,81):
        y_pred_dy = [1 if lb>= i/100 else 0 for lb in glo_arr_probs]

        mcc= metrics.matthews_corrcoef(glo_arr_labels, y_pred_dy)
        if mcc>maxmcc:
            thr=i/1000
            maxmcc=mcc
            tn, fp, fn, tp = metrics.confusion_matrix(glo_arr_labels, y_pred_dy).ravel()
            maxacc=metrics.accuracy_score(glo_arr_labels, y_pred_dy)
            balanceacc=metrics.balanced_accuracy_score(glo_arr_labels, y_pred_dy)

            maxsn = tp / (tp + fn)
            maxsp = tn / (tn + fp)
            maxf1 = 2 * tp / (2 * tp + fp + fn)
            maxyouden = maxsn + maxsp - 1
    print('acc ',maxacc)
    print('balanced_accuracy ', balanceacc)
    print('MCC ', maxmcc)
    print('sensitivity ', maxsn)
    print('specificity ', maxsp)
    print('f1score ', maxf1)
    print('youden ', maxyouden)
    print('auc',auc)
    print('thr',thr)
    return thr


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(1)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    print("Training K Fold\n")
    lst_path_train_all, lst_label_train_all,test_path_all,test_label_all = create_list_train_test()

    threshold=training_k_fold()

    print('completed')