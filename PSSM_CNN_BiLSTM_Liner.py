
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn import metrics



import torch.functional as F

from utility import save_prob_label


def create_list_train_test():

    f = open('Dataset/orderafp')
    positive_all = f.readlines()
    f.close()
    random.shuffle(positive_all)

    f = open('Dataset/ordernon_afp9493')
    negative_all = f.readlines()
    f.close()
    random.shuffle(negative_all)
    lst_path_positive_train =positive_all[0:300]
    lst_path_negative_train =negative_all[0:300]

    print("Positive train: ", len(lst_path_positive_train))
    print("Negative train: ", len(lst_path_negative_train))

    lst_positive_train_label = [1] * len(lst_path_positive_train)
    lst_negative_train_label = [0] * len(lst_path_negative_train)

    lst_path_train = lst_path_positive_train + lst_path_negative_train
    lst_label_train = lst_positive_train_label + lst_negative_train_label
    random.seed(1)
    random.shuffle(lst_path_train)
    random.seed(1)
    random.shuffle(lst_label_train)
    test_positive=positive_all[300:]
    test_negative=negative_all[300:]

    test_positive_label= [1] * len(test_positive)
    test_negative_label = [0] * len(test_negative)

    test_path_data=test_positive+test_negative
    test_label=test_positive_label+test_negative_label

    random.seed(1)
    random.shuffle(test_path_data)
    random.seed(1)
    random.shuffle(test_label)
    return lst_path_train, lst_label_train,test_path_data,test_label

def softmax(X,length,actuallength):
    s =torch.exp(X)  #batchsize,L,D
    tsum=torch.sum(s,dim=2, keepdim=False)#batchsize,L
    paddinglength=torch.unsqueeze((length - actuallength), dim=1)
    tsum=tsum-paddinglength
    tsum=torch.unsqueeze(tsum, dim=2)
    return s/tsum

class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(dim_k))
    def forward(self, x,actulength):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        #atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        #atten = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        atten = softmax(torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact,x.shape[1],actulength)
        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output

def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature0 = []
    train_y = []

    for data in batch_traindata:
        feature0.append(data[0])
        train_y.append(data[1])
    data_length = [len(data) for data in feature0]
    feature0 = torch.nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    return feature0,torch.tensor(train_y, dtype=torch.long),torch.tensor(data_length)

class BioinformaticsDataset(Dataset):
    # X: list cac file (full path)
    # Y: list label [0, 1]; 0: negative, 1: positive
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
        label = self.Y[index]

        df2= pd.read_csv('midData/PSSM_ORI_20/' + self.X[index], header=None)
        dat2 = df2.values.astype(float).tolist()

        return torch.tensor(dat2),label
    def __len__(self):
        return len(self.X)

class ResidualModule(nn.Module):
    def __init__(self,featuredim):
        super(ResidualModule,self).__init__()
        self.bn=nn.BatchNorm1d(num_features=featuredim)
        self.relu=nn.ReLU()
        self.cnn=nn.Conv1d(featuredim,featuredim,kernel_size=3,padding='same')

        self.bn1 = nn.BatchNorm1d(num_features=featuredim)
        self.relu1 = nn.ReLU()
        self.cnn1 = nn.Conv1d(featuredim, featuredim, kernel_size=3, padding='same')

    def forward(self, x):
        tmp=x
        x=self.bn(x)
        x=self.relu(x)
        x=self.cnn(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.cnn1(x)

        return tmp+x

class DCTModule(nn.Module):
    def __init__(self):
        super(DCTModule, self).__init__()
        self.cnn1 = nn.Conv1d(20, 64, 3)
        self.relu = nn.ReLU()
        self.pool1 = nn.AvgPool1d(3)

        self.cnn2 = nn.Conv1d(64, 256, 3)
        self.pool2 = nn.AvgPool1d(3)

        self.cnn3 = nn.Conv1d(256, 128, 3)
        self.pool3 = nn.AvgPool1d(3)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        self.fc1 = nn.Linear(2 * 64, 256)
        self.fc_drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.fc_drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, pssm_feature,data_length):
        pssm = pssm_feature.permute(0, 2, 1)

        pssm = self.cnn1(pssm)
        pssm = self.relu(pssm)
        ddl = data_length - 2
        pssm = self.pool1(pssm)
        ddl = torch.floor((ddl - 3) / 3) + 1

        pssm = self.cnn2(pssm)
        ddl = ddl - 2
        pssm = self.relu(pssm)
        ddl = torch.floor((ddl - 3) / 3) + 1
        pssm = self.pool2(pssm)

        pssm = self.cnn3(pssm)
        ddl = ddl - 2
        pssm = self.relu(pssm)


        pssm = pssm.permute(0, 2, 1)
        pssm = torch.nn.utils.rnn.pack_padded_sequence(pssm, ddl.to('cpu'), batch_first=True)
        pssm, (h_n, h_c) = self.lstm(pssm)

        # print('h_n[-1] shape',h_n[-1].shape)   #[8*256]
        output = torch.cat((h_n[-1], h_n[-2]), dim=1)
        # print("output fc: ", output.size())
        output = self.fc1(output)
        output = self.fc_drop1(output)
        output = self.fc2(output)
        output = self.fc_drop2(output)

        output = self.fc3(output)

        return output


def train():
    X_train = [item.strip() for item in lst_path_train_all]
    Y_train = lst_label_train_all

    epochs=40
    model = DCTModule()

    model = model.to(device)
    train_set = BioinformaticsDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                              persistent_workers=True,
                              collate_fn=coll_paddding)
    best_val_loss = 300
    crloss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for i in range(epochs):
        # adjust_learning_rate(optimizer, i, 0.1)

        epoch_loss_train = 0.0
        nb_train = 0
        for data_pssm_x, data_y, length in train_loader:
            optimizer.zero_grad()
            y_pred = model(data_pssm_x.to(device), length.to(device))
            data_y = data_y.to(device)

            single_loss = crloss(y_pred, data_y.data)

            single_loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + single_loss.item()
            nb_train += 1
        epoch_loss_avg = epoch_loss_train / nb_train
        if best_val_loss > epoch_loss_avg:
            model_fn = "protein_atp01.pkl"
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            if i % 10 == 0:
                print('epochs ', i)
                print("Save model, best_val_loss: ", best_val_loss)
def test():
    X_test = [item.strip() for item in test_path_all]
    test_set = BioinformaticsDataset(X_test, test_label_all)

    test_load = DataLoader(dataset=test_set, batch_size=64,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    model = DCTModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load('protein_atp01.pkl'))
    model.eval()

    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []

    with torch.no_grad():
        for data_pssm_x, data_y, length in test_load:
            y_pred= model(data_pssm_x.to(device), length.to(device))
            y_pred=torch.nn.functional.softmax(y_pred,dim=1)
            arr_probs.extend(y_pred[:, 1].to('cpu'))
            y_pred=torch.argmax(y_pred, dim=1).to('cpu')
            #y_pred = (y_pred[:, 1] > thredhold).long().to('cpu')

            arr_labels.extend(data_y)
            arr_labels_hyps.extend(y_pred)

    print('-------------->')

    auc =metrics.roc_auc_score(arr_labels, arr_probs)
    print('acc ', metrics.accuracy_score(arr_labels, arr_labels_hyps))
    print('balanced_accuracy ', metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps))
    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    print('tn, fp, fn, tp ',tn, fp, fn, tp )
    print('MCC ', metrics.matthews_corrcoef(arr_labels, arr_labels_hyps))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('<----------------')
    save_prob_label(arr_probs, arr_labels, 'PSSM-CNN-Bi-LSTM.csv')
    print('<----------------save to csv finish')

if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    lst_path_train_all, lst_label_train_all, test_path_all, test_label_all = create_list_train_test()
    train()
    test()
    print('completed')

