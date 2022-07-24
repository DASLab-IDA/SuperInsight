import joblib

import os

import csv

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import numpy as np

import torch

import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from torch.optim import Adam

import torch.nn.functional as F

import torch.nn as nn

import torch.utils.data as Data


class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        self.hidden1 = nn.Linear(in_features=30, out_features=16, bias=True)
        self.hidden2 = nn.Linear(16, 8)
        self.hidden3 = nn.Linear(8, 4)
        self.predict = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]


def train(X_train, y_train):
    if os.path.exists("all label trained models/neural_network_model_corpus.m"):
        mlpreg = torch.load('all label trained models/neural_network_model_corpus.m')
    else:
        train_xt = torch.from_numpy(X_train.astype(np.float32))
        train_yt = torch.from_numpy(y_train.astype(np.float32))
        train_data = Data.TensorDataset(train_xt, train_yt)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=1)
        mlpreg = MLPregression()
        print(mlpreg)
        optimzer = Adam(mlpreg.parameters(), lr=0.0001)
        loss_func = nn.MSELoss()
        train_loss_all = []
        for epoch in range(300):
            train_loss = 0
            train_num = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                output = mlpreg(b_x)
                # print("output:", end=' ')
                # print(output)
                loss = loss_func(output, b_y)
                # print("loss:",end=' ')
                # print(loss)
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
                train_loss += loss.item() * b_x.size(0)
                train_num += b_x.size(0)
                # print("train_loss:", end=' ')
                # print(train_loss)
                # print("train_num:", end=' ')
                # print(train_num)
            train_loss_all.append(train_loss / train_num)
            print("epoch:", end=' ')
            print(epoch, end=', ')
            print("loss:", end=' ')
            print(train_loss / train_num)
        torch.save(mlpreg, 'all label trained models/neural_network_model_corpus.m')
    return mlpreg


def load_model():
    if os.path.exists("all label trained models/neural_network_model_corpus.m"):
        reg = torch.load('all label trained models/neural_network_model_corpus.m')
    else:
        reg = None
    return reg


def predict(mlpreg: object, X_test):
    test_xt = torch.from_numpy(X_test.astype(np.float32))
    pre_y = mlpreg(test_xt)
    pre_y = pre_y.data.numpy()
    return pre_y


if __name__ == '__main__':
    train_dataset = ["toy_dataset", "car_sales", "hotel_booking"]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X = []
    y = []
    cate_X = []
    All_X = []
    for i in range(len(train_dataset)):
        f_r = open(f'dataset_example/test_featured_{train_dataset[i]}_fifo.csv', 'r', encoding='utf-8')
        csv_reader = csv.reader(f_r)
        for row in tqdm(csv_reader):
            y.append(float(row[3]))
            row[4] = row[4].replace("\n", "")
            row[4] = row[4].replace("[", "")
            row[4] = row[4].replace("]", "")
            row[4] = row[4].strip()
            # print(row[5])
            row[4] = ' '.join(row[4].split())
            single_data = []
            cate_data = []
            data_arr = row[4].split(" ")
            category_idx = [0, 2]
            count = 0
            for data in data_arr:
                if count in category_idx:
                    cate_data.append(int(eval(data)))
                else:
                    single_data.append(float(data))
                count += 1
            X.append(single_data)
            cate_X.append(cate_data)

        for i in range(len(X)):
            X[i] = np.array(X[i]).astype('float64')
        f_r.close()


    new_X = X
    new_cate_X = cate_X
    new_y = y
    # data standardization
    scale = StandardScaler()
    new_X = scale.fit_transform(new_X)
    for i in range(len(new_X)):
        l = new_X[i].tolist()
        l.extend(new_cate_X[i])
        All_X.append(l)
    X_train, X_test, y_train, y_test = train_test_split(All_X, new_y, test_size=0.3, random_state=2022)
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')
    y_train = np.array(y_train).astype('float32')
    model = train(X_train, y_train)
    X_test = np.array(X_test).astype('float64')
    y_predict = predict(model, X_test)
    for i in range(len(y_predict)):
        print(y_test[i], end=' , ')
        print(y_predict[i])
    print(mean_squared_error(y_test, y_predict))