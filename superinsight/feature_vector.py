from functools import lru_cache
from metainsight.insight import *
import numpy as np
from config import *
from statsmodels import robust
from math import log
from scipy import stats
from data_source import *
from explore import *
import yaml
import json


global data_source

global column_meta_data
global extend_list
global aggregate_function
global pattern_dict

def get_default_config():
    with open("default_config.yml", "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        global extend_list
        extend_list = data['extend_list']
        global aggregate_function
        aggregate_function = data['aggregate_function']
        global pattern_dict
        pattern_dict = data['pattern_dict']
        return data


def get_column_meta_data(dataset):
    with open(f"dataset_example/config/{dataset}.json", 'r', encoding='utf8') as fp:
        json_data = json.load(
            fp)  # {'Year': 'Temporal', 'Brand': 'Categorical', 'Category': 'Categorical', 'Model': 'Categorical', 'Sales': 'Numerical'}
        global column_meta_data
        column_meta_data = {
            dataset: json_data}  # {'CarSales': {'Year': 'Temporal', 'Brand': 'Categorical', 'Category': 'Categorical', 'Model': 'Categorical', 'Sales': 'Numerical'}}
        return column_meta_data


get_default_config()
get_column_meta_data(dataset)

def feature_data_source(db):
    global data_source
    data_source = db


@lru_cache(maxsize=None)
def get_sql_result(filter_param: str = None):
    global data_source
    data_source = DataSource(host=host, user=user, password=password, database=database, num_thread=1)
    columns = []
    global column_meta_data
    for key in column_meta_data[dataset]:
        if column_meta_data[dataset][key] == "Numerical":
            columns.append(key)
    column_str = ", ".join(columns)
    sql = f"SELECT {column_str} FROM {dataset}"
    if filter_param is not None:
        sql = sql + f" WHERE {filter_param}"
    sql = sql + ";"
    sql_res = data_source.execute_sql(sql)
    res = {}

    for column in columns:
        res[column] = []

    for record in sql_res:
        for column in columns:
            if record[column] is not None:
                res[column].append(record[column])

    return res


@lru_cache(maxsize=None)
def get_agg_result(filter_param: str = None):
    global data_source
    data_source = DataSource(host=host, user=user, password=password, database=database, num_thread=1)
    num_columns = []
    cat_columns = []
    global column_meta_data
    for key in column_meta_data[dataset]:
        if column_meta_data[dataset][key] == "Numerical":
            num_columns.append(key)
        else:
            cat_columns.append(key)
    agg_functions = ["MAX", "MIN", "SUM", "AVG", "VARIANCE"]
    agg_list = []

    for column in num_columns:
        for agg in agg_functions:
            agg_list.append(agg + "(" + column + ")")
    for column in cat_columns:
        agg_list.append(f"COUNT(DISTINCT {column})")
    agg_list.append("COUNT(*)")
    agg_str = ", ".join(agg_list)
    sql = f"SELECT {agg_str} FROM {dataset}"
    if filter_param is not None:
        sql = sql + f" WHERE {filter_param}"
    sql = sql + ";"
    sql_res = data_source.execute_sql(sql)
    # print(sql_res)
    return sql_res[0]


def cal_entropy(nums: int, labelCounts: dict):
    entropy_value = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / nums
        entropy_value -= prob * log(prob, 2)
    return entropy_value


def cal_gini(nums: int, labelCounts: dict):
    g_value = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / nums
        g_value -= prob * prob
    return g_value


def cal_outlier_25(nums: int, labelCounts: dict, target: int):
    count = 0
    for key in labelCounts:
        if key< target:
            count += labelCounts[key]
    return float(count)/nums


def cal_outlier_75(nums: int, labelCounts: dict, target: int):
    count = 0
    for key in labelCounts:
        if key > target:
            count += labelCounts[key]
    return float(count)/nums


@lru_cache(maxsize=None)
def calculate_feature(column: str, subspace: str = None):
    column_type = column_meta_data[dataset][column]

    if column_type == "Categorical" or column_type == "Temporal":
        feature = np.zeros([1, 5])
    else:
        feature = np.zeros([1, 26])
    if column_type == "Categorical" or column_type == "Temporal":
        # is temporal, is categorical, # of distinct value
        if column_type == "Temporal":
            feature[0][0] = 1
        else:
            feature[0][1] = 1
        feature[0][2] = get_agg_result(subspace)["COUNT(*)"] / get_agg_result()["COUNT(*)"]
        feature[0][3] = get_agg_result(subspace)[f"COUNT(DISTINCT {column})"]
        feature[0][4] = get_agg_result()[f"COUNT(DISTINCT {column})"]
    else:
        sql_result = get_sql_result(subspace)[column]
        # np_y = np.array(sql_result)
        np_y = np.array(sql_result).astype(np.float)
        nums = len(np_y)
        label_counts = {}
        for v in np_y:
            if v not in label_counts.keys():
                label_counts[v] = 0
            label_counts[v] += 1
        agg_result = get_agg_result(subspace)
        # feature[0][0] = np.mean(np_y)  # mean
        feature[0][0] = agg_result[f"AVG({column})"]
        feature[0][1] = np.median(np_y)  # median
        # feature[0][2] = np.var(np_y)  # variance
        feature[0][2] = agg_result[f"VARIANCE({column})"]
        feature[0][3] = np.std(np_y)  # standard deviation
        if feature[0][0] != 0:
            feature[0][4] = feature[0][3] / feature[0][0]  # coefficient of variance
        # feature[0][5] = np.min(np_y)  # minimum
        # feature[0][6] = np.max(np_y)  # maximum
        feature[0][5] = agg_result[f"MIN({column})"]
        feature[0][6] = agg_result[f"MAX({column})"]
        feature[0][7] = np.percentile(np_y, (25))  # 25th percentile
        feature[0][8] = np.percentile(np_y, (75))  # 75th percentile
        feature[0][9] = robust.mad(np_y)  # median absolute deviation
        # feature[0][10] = sum([abs(np_y[i] - feature[0][0]) for i in range(len(np_y))]) / len(np_y)  # average absolute deviation
        feature[0][11] = 0  # what is the quantitative coefficient of dispersion?
        feature[0][12] = cal_entropy(nums, label_counts)  # entropy
        feature[0][13] = cal_gini(nums, label_counts)  # Gini
        feature[0][14] = stats.skew(np_y)  # skewness
        feature[0][15] = stats.kurtosis(np_y)  # kurtosis
        feature[0][16] = stats.moment(np_y, moment=5)
        feature[0][17] = stats.moment(np_y, moment=6)
        feature[0][18] = stats.moment(np_y, moment=7)
        feature[0][19] = stats.moment(np_y, moment=8)
        feature[0][20] = stats.moment(np_y, moment=9)
        feature[0][21] = stats.moment(np_y, moment=10)
        # ValueError: skewtest is not valid with less than 8 samples; 7 samples were given.
        if len(np_y) > 7:
            p_value = stats.normaltest(np_y)[1]
            if p_value < 0.01:
                feature[0][22] = True  # p<0.01
                feature[0][23] = True  # p<0.05
            elif p_value < 0.05:
                feature[0][22] = False  # p>0.01
                feature[0][23] = True  # p<0.05
            else:
                feature[0][22] = False  # p>0.01
                feature[0][23] = False  # p>0.05
        feature[0][24] = cal_outlier_25(nums, label_counts, feature[0][7])
        feature[0][25] = cal_outlier_75(nums, label_counts, feature[0][8])
        # print(feature[0])
        # pass
    return feature


@lru_cache(maxsize=None)
def calculate_feature_with_category(column: str, subspace: str = None):
    column_type = column_meta_data[dataset][column]
    if column_type == "Categorical" or column_type == "Temporal":
        feature = np.zeros([1, 2])
    else:
        feature = np.zeros([1, 26])
    if column_type == "Categorical" or column_type == "Temporal":
        # is temporal, is categorical, # of distinct value
        if column_type == "Temporal":
            # feature[0][0] = "Temporal"
            feature[0][0] = 0
        else:
            # feature[0][0] = "Categorical"
            feature[0][0] = 1
        feature[0][1] = get_agg_result(subspace)["COUNT(*)"] / get_agg_result()["COUNT(*)"]

    else:
        sql_result = get_sql_result(subspace)[column]
        # np_y = np.array(sql_result)
        np_y = np.array(sql_result).astype(np.float)
        nums = len(np_y)
        label_counts = {}
        for v in np_y:
            if v not in label_counts.keys():
                label_counts[v] = 0
            label_counts[v] += 1
        agg_result = get_agg_result(subspace)
        if len(np_y) > 0:
            feature[0][0] = agg_result[f"AVG({column})"]
            feature[0][1] = np.median(np_y)  # median
            feature[0][2] = agg_result[f"VARIANCE({column})"]
            feature[0][3] = np.std(np_y)  # standard deviation
            if feature[0][0] != 0:
                feature[0][4] = feature[0][3] / feature[0][0]  # coefficient of variance
            feature[0][5] = agg_result[f"MIN({column})"]
            feature[0][6] = agg_result[f"MAX({column})"]
            feature[0][7] = np.percentile(np_y, (25))  # 25th percentile
            feature[0][8] = np.percentile(np_y, (75))  # 75th percentile
            feature[0][9] = robust.mad(np_y)  # median absolute deviation
            # feature[0][10] = sum([abs(np_y[i] - feature[0][0]) for i in range(len(np_y))]) / len(np_y)  # average absolute deviation
            feature[0][11] = 0  # what is the quantitative coefficient of dispersion?
            feature[0][12] = cal_entropy(nums, label_counts)  # entropy
            feature[0][13] = cal_gini(nums, label_counts)  # Gini
            feature[0][14] = stats.skew(np_y)  # skewness
            feature[0][15] = stats.kurtosis(np_y)  # kurtosis
            feature[0][16] = stats.moment(np_y, moment=5)
            feature[0][17] = stats.moment(np_y, moment=6)
            feature[0][18] = stats.moment(np_y, moment=7)
            feature[0][19] = stats.moment(np_y, moment=8)
            feature[0][20] = stats.moment(np_y, moment=9)
            feature[0][21] = stats.moment(np_y, moment=10)
            # ValueError: skewtest is not valid with less than 8 samples; 7 samples were given.
            if len(np_y) > 7:
                p_value = stats.normaltest(np_y)[1]
                if p_value < 0.01:
                    feature[0][22] = True  # p<0.01
                    feature[0][23] = True  # p<0.05
                elif p_value < 0.05:
                    feature[0][22] = False  # p>0.01
                    feature[0][23] = True  # p<0.05
                else:
                    feature[0][22] = False  # p>0.01
                    feature[0][23] = False  # p>0.05
            feature[0][24] = cal_outlier_25(nums, label_counts, feature[0][7])
            feature[0][25] = cal_outlier_75(nums, label_counts, feature[0][8])
    return feature
