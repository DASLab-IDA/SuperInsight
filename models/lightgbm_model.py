import lightgbm as lgb

import joblib

import os

import csv

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

import pandas as pd

import numpy as np


def train(X_train: list, y_train: list,X_test: list, y_test: list):
    if os.path.exists("all label trained models/lightgbm_model_corpus.m"):
        model = joblib.load('all label trained models/lightgbm_model_corpus.m')
    else:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=100)
        model = lgb.train(params, lgb_train, num_boost_round=1000)
        joblib.dump(model, 'all label trained models/lightgbm_model_corpus.m')
    return model


def train_with_category(X_train: list, y_train: list):
    if os.path.exists("all label trained models/lightgbm_model_corpus.m"):
        model = joblib.load('all label trained models/lightgbm_model_corpus.m')
    else:
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1
        }
        df_train = pd.DataFrame(X_train, columns=['f1', 'f2',
                                                  'f3', 'f4', 'f5', 'f6',
                                                  'f7', 'f8', 'f9', 'f10',
                                                  'f11', 'f12', 'f13', 'f14',
                                                  'f15', 'f16', 'f17', 'f18',
                                                  'f19', 'f20', 'f21', 'f22',
                                                  'f23', 'f24', 'f25', 'f26',
                                                  'f27', 'f28', 'c1', 'c2'])
        lgb_train = lgb.Dataset(df_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=100)
        model = lgb.train(params, lgb_train, num_boost_round=1000, categorical_feature=['c1', 'c2'])
        joblib.dump(model, 'all label trained models/lightgbm_model_corpus.m')
    return model


def load_model():
    if os.path.exists("all label trained models/lightgbm_model_corpus.m"):
        estimator = joblib.load('all label trained models/lightgbm_model_corpus.m')
    else:
        estimator = None
    return estimator


def predict(reg: object, X_test: list):
    return reg.predict(X_test, num_iteration=reg.best_iteration)



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
    model = train_with_category(X_train, y_train)
    y_predict = predict(model, X_test)
    for i in range(len(y_predict)):
        print(y_test[i], end=' , ')
        print(y_predict[i])
    print(mean_squared_error(y_test, y_predict))