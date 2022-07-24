from sklearn.ensemble import RandomForestRegressor

import joblib

import os

import csv

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

import numpy as np


def train(X_train: list, y_train: list):
    if os.path.exists("all label trained models/random_forest_model_corpus.m"):
        estimator = joblib.load('all label trained models/random_forest_model_corpus.m')
    else:
        estimator = RandomForestRegressor(random_state=2022, n_estimators=100)
        estimator.fit(X_train, y_train)
        joblib.dump(estimator, 'all label trained models/random_forest_model_corpus.m')
    return estimator


def load_model():
    if os.path.exists("all label trained models/random_forest_model_corpus.m"):
        estimator = joblib.load('all label trained models/random_forest_model_corpus.m')
    else:
        estimator = None
    return estimator


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

        for j in range(len(X)):
            X[j] = np.array(X[j]).astype('float64')
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