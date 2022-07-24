from sklearn.neighbors import KNeighborsRegressor

import joblib

import os

import csv

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import numpy as np

from sklearn.preprocessing import StandardScaler


def train(X_train: list, y_train: list, k: int = 3):
    # the parameter k can be set by yourself, the default is 3
    if os.path.exists("all label trained models/knn_model_corpus.m"):
        reg = joblib.load('all label trained models/knn_model_corpus.m')
    else:
        reg = KNeighborsRegressor(n_neighbors=k)
        reg.fit(X_train, y_train)
        joblib.dump(reg, "all label trained models/knn_model_corpus.m")
    return reg


def load_model():
    if os.path.exists("all label trained models/knn_model_corpus.m"):
        reg = joblib.load('all label trained models/knn_model_corpus.m')
    else:
        reg = None
    return reg


def predict(reg: object, X_test: list):
    return reg.predict(X_test)


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

    model = train(X_train, y_train)
    X_test = np.array(X_test).astype('float64')

    y_predict = predict(model, X_test)
    for i in range(len(y_predict)):
        print(y_test[i], end=' , ')
        print(y_predict[i])
    print(mean_squared_error(y_test, y_predict))
