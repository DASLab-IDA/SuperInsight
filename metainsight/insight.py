import csv
import functools
import queue
import sys
import threading
import time
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.stattools import acf
from config import *
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from kneed import KneeLocator
from scipy import signal
import warnings
from explore import *
import yaml
import json


global column_meta_data
global extend_list
global aggregate_function
global pattern_dict
global dataset


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


warnings.filterwarnings("ignore")
distinct_dict = {}
measure_list = []
record_num = 0
# data_source_list = queue.Queue()
global data_source

lock = threading.Lock()

sql_exe_count = 0
sql_call_count = 0
pattern_exe_count = 0
pattern_call_count = 0

sql_time = 0.0
pattern_time = 0.0
is_augmented = True


# global dataset
#
#
# def set_dataset(data_set):
#     global dataset
#     dataset = data_set


def insight_data_source(db, num=1):
    global data_source
    data_source = db


def set_augment():
    global is_augmented
    is_augmented = True


def time_consume():
    return sql_time, pattern_time


def sql_cache_consume():
    return sql_call_count, sql_exe_count


def pattern_cache_consume():
    return pattern_call_count, pattern_exe_count


def pattern_evaluate_outstanding_1(result_x, result_y, pattern_list):
    #  "outstanding_#1": subspace with the highest aggregate value
    """
    Calculate whether there is an outstanding_1 mode in the data_scope
    (if a value accounts for more than 50% of the total value or a value is more than 2 times the average value,
    the value is considered to have the outstanding_1 feature)
    """
    max_value = min(result_y)
    max_index = -1
    for k in range(len(result_y)):
        if result_y[k] > max_value:
            max_value = result_y[k]
            max_index = k
    if max_index != -1:
        for k in range(len(result_y)):
            if result_y[k] == max_value:
                if result_y[k] >= 2 * cal_list_mean(result_y) or result_y[k] / sum(
                        result_y) >= 0.5:
                    pattern_list["outstanding_#1"] = result_x[k]
    return pattern_list


def evaluate_pattern_outstanding_last(result_x, result_y, pattern_list):
    """
    Calculate whether there is an outstanding_last mode in the data_scope
    (if a value is less than or equal to 1/2 of the average,
    the value is considered to have the outstanding_last feature)
    """
    min_value = max(result_y)
    min_index = -1
    for k in range(len(result_y)):
        if result_y[k] < min_value:
            min_value = result_y[k]
            min_index = k
    if min_index != -1:
        for k in range(len(result_y)):
            if result_y[k] == min_value:
                if result_y[k] <= cal_list_mean(result_y) / 2:
                    pattern_list["outstanding_#last"] = result_x[k]
    return pattern_list


def evaluate_pattern_attribution(result_x, result_y, pattern_list):
    """
    Calculate whether there is an attribute in the data_scope
    (if a value is greater than 50% of the total value, the value is considered to have an attribute)
    """
    sum_result_y = sum(result_y)
    attr_index = -1
    for k in range(len(result_y)):
        if result_y[k] / sum_result_y >= 0.5:
            attr_index = k
            break
        else:
            continue
    if attr_index != -1:
        pattern_list["attribution"] = result_x[attr_index]
    return pattern_list


def evaluate_pattern_change_point(result_x, result_y, pattern_list):
    """
    """
    # x, y = x.tolist(), y.tolist()
    Xi = [x for x in range(len(result_y))]
    output_knees = []
    for curve in ['convex', 'concave']:
        for direction in ['increasing', 'decreasing']:
            model = KneeLocator(x=Xi, y=result_y, curve=curve, direction=direction, online=False)
            if model.knee != Xi[0] and model.knee != Xi[-1] and model.knee is not None:
                # output_knees.append((model.knee, model.knee_y, curve, direction))
                output_knees.append(result_x[model.knee])

        if output_knees.__len__() != 0:
            pattern_list["change_point"] = output_knees
    return pattern_list


def evaluate_pattern_trend(result_x, result_y, pattern_list):
    """
    Calculate whether the data_scope has a certain trend, and after smoothing the data,
    determine whether the data is (non-strict) monotonically increasing (or decreasing)
    """
    smooth_result_y = pd.DataFrame({'Smooth': result_y}).ewm(alpha=0.8).mean()['Smooth'].values.tolist()
    precision = 1e-6
    peaks = signal.argrelextrema(np.array(smooth_result_y), np.greater)[0]
    # print(peaks)
    troughs = signal.argrelextrema(np.array(smooth_result_y), np.less)[0]
    # print(troughs)
    if len(peaks) == 0 and len(troughs) == 0:
        index = 0
        while (index + 1) < len(result_y) and abs(smooth_result_y[index] - smooth_result_y[index + 1]) <= precision:
            index = index + 1
        if index == len(result_y) - 1:
            pattern_list["evenness"] = "True"
            pattern_list["seasonality"] = f"周期值：{1}"
            return pattern_list
        if index < len(result_y) - 1:
            if len(result_y) > 1 and smooth_result_y[index + 1] > smooth_result_y[index]:
                pattern_list["trend"] = "increase"
            else:
                pattern_list["trend"] = "decrease"

    if len(peaks) == 1 and len(troughs) == 0:
        pattern_list["unimodality"] = "reverse-U_type"
        if "outstanding_#1" not in pattern_list.keys():
            pattern_list["outstanding_#1"] = result_x[peaks[0]]
    if len(peaks) == 0 and len(troughs) == 1:
        pattern_list["unimodality"] = "U_type"
        if "outstanding_#last" not in pattern_list.keys():
            pattern_list["outstanding_#last"] = result_x[troughs[0]]
    return pattern_list


def evaluate_pattern_seasonality(result_x, result_y, pattern_list):
    """
    Determine whether the data_scope is periodic
    (use Fourier transform to calculate the period candidate value,
     and then calculate the autocorrelation coefficient according to the period candidate value,
    the largest one of the autocorrelation values and the number of periods corresponding to
     the autocorrelation coefficient whose value is greater than or equal to 0.5 is period for data_scope)
    """
    maxperiod, maxvalue = fft(result_y)
    if maxperiod != -1:
        if maxvalue > 0.5:
            pattern_list["seasonality"] = f"周期值：{maxperiod}"
    return pattern_list


def evaluate_pattern_outlier(result_x, result_y, pattern_list):
    """
    Calculate whether there are outliers in data_scope
    (the value is greater than or equal to the mean + 2.5*standard deviation or
     the value is less than or equal to the mean-2.5*standard deviation, the point is considered an outlier)
    """
    # in this case,we assume that the result_y obey the normal distribution
    # if x > mean(result_y)+2.5*std(result_y) or x < mean(result_y)-2.5*std(result_y)
    # we regard x as a outlier
    outlier_list = []
    clf = IsolationForest()
    list_y = list(map(lambda x: [x], result_y))
    clf.fit(list_y)
    y_pred = clf.predict(list_y)
    # print(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] != 1:
            outlier_list.append(result_x[i])
    if len(outlier_list) != 0:
        pattern_list["outlier"] = outlier_list
    return pattern_list


def evaluate_pattern_evenness(result_x, result_y, pattern_list):
    """
    Calculate whether the data_scope data are approximately equal
    (if the standard deviation/mean <= 0.03, this group of data is considered to be approximately equal)
    """
    if len(result_y) >= 4:
        dftest = adfuller(result_y, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        if dfoutput['Test Statistic'] < dfoutput['Critical Value (1%)'] and \
                dfoutput['Test Statistic'] < dfoutput['Critical Value (5%)'] and \
                dfoutput['Test Statistic'] < dfoutput['Critical Value (10%)']:
            pattern_list["evenness"] = "True"
    return pattern_list


type_to_function = {"outstanding_1": pattern_evaluate_outstanding_1,
                    "outstanding_last": evaluate_pattern_outstanding_last, "attribution": evaluate_pattern_attribution,
                    "change_point": evaluate_pattern_change_point, "trend": evaluate_pattern_trend,
                    "seasonality": evaluate_pattern_seasonality,
                    "outlier": evaluate_pattern_outlier, "evenness": evaluate_pattern_evenness}


def call_evaluate_function(pattern_type: str):
    return type_to_function[pattern_type]


def write_down_exception(e, filter_param, breakdown, measure):
    with open('test/sql_error.csv', 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([filter_param, breakdown, measure])
    print(f"=======e:{filter_param}, {breakdown}, {measure}")
    print(e.with_traceback())
    sys.exit()


@lru_cache(maxsize=None)
def evaluate_pattern(measure: str, filter_param: str, breakdown: str = None):
    """
    evaluate data scope's patterns using lru cache
    :param measure:
    :param filter_param:
    :param breakdown:
    :return:
    """
    global pattern_time
    global sql_call_count
    global pattern_exe_count
    global pattern_dict
    pattern_list = {}
    pattern_exe_count = pattern_exe_count + 1
    sql_call_count = sql_call_count + 1
    x, y = execute_sql(measure, filter_param, breakdown, is_augmented)
    pattern_start_time = time.perf_counter()
    try:
        if len(y) > 1:
            pattern_list = call_evaluate_function("trend")(x, y, pattern_list)
            if "evenness" not in pattern_list.keys():
                # pattern_list = pattern_evaluate_outstanding_1(x, y, pattern_list)
                # pattern_list = evaluate_pattern_outstanding_last(x, y, pattern_list)
                # pattern_list = evaluate_pattern_attribution(x, y, pattern_list)
                # pattern_list = evaluate_pattern_change_point(x, y, pattern_list)
                # pattern_list = evaluate_pattern_outlier(x, y, pattern_list)
                # pattern_list = evaluate_pattern_evenness(x, y, pattern_list)
                global dataset
                for pattern_type in pattern_dict[column_meta_data[dataset][breakdown]]:
                    if pattern_type == "trend":
                        continue
                    pattern_list = call_evaluate_function(pattern_type)(x, y, pattern_list)

            # print("pattern_list======================", end='')
            # print(pattern_list)
    except Exception as e:
        write_down_exception(e, filter_param, breakdown, measure)

    # print("pattern execute count:",end=' ')
    # print(pattern_exe_count,end=' ** ')
    # print("pattern call count:",end=' ')
    # print(pattern_call_count)
    pattern_end_time = time.perf_counter()
    pattern_time = pattern_time + pattern_end_time - pattern_start_time
    return pattern_list


def sql_result(sql):
    global sql_time
    global sql_exe_count
    global data_source

    sql_exe_count = sql_exe_count + 1

    sql_start_time = time.perf_counter()
    res = data_source.execute_sql(sql)
    sql_end_time = time.perf_counter()

    sql_time = sql_time + sql_end_time - sql_start_time

    return res


@lru_cache(maxsize=None)
def normal_sql(measure: str, filter_param: str, breakdown: str):
    global dataset
    try:
        if breakdown is not None:
            sql = "select {},{} from {} where {} group by {} order by {} asc;".format(breakdown, measure, dataset,
                                                                                      filter_param, breakdown,
                                                                                      breakdown)
            # print("Query:" + sql)
            res = sql_result(sql)
        else:
            sql = "select {} from {} where {};".format(measure, dataset, filter_param)
            # print(sql)
            res = sql_result(sql)
    except Exception as e:
        write_down_exception(e, filter_param, breakdown, measure)
    # print("sql execute count:",end=' ')
    # print(sql_exe_count,end=' ** ')
    # print("sql call count:",end=' ')
    # print(sql_call_count)
    return res


@lru_cache(maxsize=None)
def augment_sql(filter_param: str, breakdown: str, measure:str):
    global dataset
    try:
        agg_list = []
        # for measure_col, measure_col_type in column_meta_data[dataset].items():
        #     if column_type_list[measure_col_type] == "Numerical" and measure_col != breakdown:
        measure_col = measure[4:-1]
        for agg in aggregate_function:
            agg_list.append("{}({})".format(agg, measure_col))

        agg_measure = ",".join(agg_list)
        if breakdown is not None:
            sql = "select {},{} from {} where {} group by {} order by {} asc;".format(breakdown, agg_measure, dataset,
                                                                                      filter_param,
                                                                                      breakdown, breakdown)
            res = sql_result(sql)
            # print(res)
        else:
            sql = "select {} from {} where {};".format(agg_measure, dataset, filter_param)
            # print(sql)
            res = sql_result(sql)
    except Exception as e:
        write_down_exception(e, filter_param, breakdown, agg_measure)

    return res


month = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
}


def execute_sql(measure: str, filter_param: str, breakdown: str = None, augmented=True):
    x = []
    y = []
    if augmented:
        res = augment_sql(filter_param, breakdown, measure)
    else:
        res = normal_sql(measure, filter_param, breakdown)

    # if breakdown on the month
    def sort_month(a, b):
        return month[a[breakdown]] - month[b[breakdown]]

    if breakdown == "arrival_date_month" and not isinstance(res, tuple):
        res.sort(key=functools.cmp_to_key(sort_month))

    # print(f"sorted_car_sales res: {res}")
    if breakdown is not None:
        for item in res:
            if item[breakdown] is not None and item[measure] is not None:
                x.append(item[breakdown])
                if isinstance(item[measure], Decimal):
                    y.append(float(item[measure]))
                else:
                    y.append(item[measure])
    else:
        y.append(res[0]["count(*)"])

    return x, y


def close_data_source():
    data_source.close()


def measure_list_generate(dataset):
    global column_meta_data
    for col, col_type in column_meta_data[dataset].items():
        if col_type == "Numerical":
            global aggregate_function
            for agg in aggregate_function:
                measure_list.append(agg + "(" + col + ")")


def get_distinct_value(dataset):
    """
    get distinct value in every column
    (to be optimized: supposed to be executed alone with other group-by sql)
    :return:
    """
    global data_source
    global column_meta_data
    for col, col_type in column_meta_data[dataset].items():
        if col_type != "Numerical":
            sql = "select distinct " + col + " from " + dataset + ";"
            print("==========execute distinct value for col:" + col)
            res = data_source.execute_sql(sql)
            print(res)
            value_list = []
            for item in res:
                value_list.append(item[col])
            distinct_dict[col] = value_list
    print(distinct_dict)


def get_record_num(dataset2):
    global data_source
    res = data_source.execute_sql("select count(*) from " + dataset2 + ";")
    global dataset
    dataset = dataset2
    # print(res)
    global record_num
    record_num = res[0]['count(*)']


def cal_list_mean(list):
    return np.mean(np.array(list))


def fft(data):
    """
    Determine the most likely period value of data and the autocorrelation coefficient corresponding to the period value
    :param data:
    :return:max_period-most likely number of cycles，max_value：The autocorrelation coefficient corresponding to the most likely number of cycles
    """
    fft_series = fft(data)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    top_k_seasons = 4
    # top K=3 index
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    # Expected time period
    score_list = []
    max_value = -1
    max_period = -1
    for lag in fft_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(list(map(float, data)), nlags=lag)[-1]
        score_list.append(acf_score)
        if acf_score > max_value:
            max_value = acf_score
            max_period = lag
    return max_period, max_value


#  convert in/float
def convert_digital_num(string):
    string1 = str(string)
    if string1.count('-') == 1 and string1[0] == '-':
        str2 = string1.split("-")[-1]
        if str2.isdigit():
            return int(string1)
        elif str2.count('.') == 1:
            left_str = str2.split(".")[0]
            right_str = str2.split(".")[1]
            if left_str.isdigit() and right_str.isdigit():
                return float(string1)
            else:
                return string
        else:
            return string
    elif string1.isdigit():
        return int(string1)
    elif string1.count('.') == 1:
        left_str = string1.split(".")[0]
        right_str = string1.split(".")[1]
        if left_str.isdigit() and right_str.isdigit():
            return float(string1)
        else:
            return string
    else:
        return string


def cross_entropy(y):
    y = np.float_(y)
    return -1 * np.sum(y * np.log(y))


class Subspace:
    def __init__(self, filter_dict: dict):
        self.filter_dict = filter_dict
        self.impact = None

    def get_impact(self):
        """
        get the impact of data scope
        :return:
        """
        if self.impact is None:
            filter_params = self.filter_param()
            if filter_params is not None:
                # sql = "select count(*) from {} where {};".format(dataset, self.filter_param())
                # # print(sql)
                # res = data_source.excute_sql(sql)
                global sql_call_count
                sql_call_count = sql_call_count + 1
                x, y = execute_sql("count(*)", self.filter_param(), None, False)
                self.impact = y[0] / record_num
                # print("IMPACT of {}: {} ".format(str(self.filter_dict), self.impact))
            else:
                self.impact = 0

        return self.impact

    # TODO: fix the bug
    def filter_param(self):
        filter_list = []
        for col, value in self.filter_dict.items():
            if value is not None:
                if isinstance(value, str):
                    filter_list.append(col + " = \"" + value + "\"")
                else:
                    filter_list.append(col + " = " + str(value))
                filter_param = " and ".join(filter_list)
                return filter_param
            else:
                return None

    def extend(self, extend_col, extend_value):
        if self.filter_dict[extend_col] == extend_value:
            return self
        else:
            new_dict = dict(self.filter_dict)
            new_dict[extend_col] = extend_value
            return Subspace(new_dict)

    def get_columns(self):
        return [key for key in self.filter_dict.keys()]

    def get_legend(self):
        vals = []
        for val in self.filter_dict.values():
            vals.append(val)
        new_vals = []
        for val in vals:
            new_vals.append(str(val))
        return "{" + ",".join((new_vals)) + "}"


class DataScope:
    def __init__(self, subspace: Subspace, breakdown: str, measure: str):
        """
        initial function of data scope
        :param subspace: the filter info of data scope
        :param breakdown: the group info of data scope
        :param measure: the aggregate info of data scope
        """
        self.subspace = subspace
        self.breakdown = breakdown
        self.measure = measure
        self.result_x = []
        self.result_y = []
        self.smooth_result_y = []
        self.pattern_list = {}

    def __lt__(self, other):
        return self.subspace.get_impact() > other.subspace.get_impact()

    def execute_data_source(self):
        """
        execute the corresponding query of data scope
        :return: the result set of query
        """
        # sql = "select {},{} from {} where {} group by {};".format(self.breakdown, self.measure, self.dataset,
        #                                                           self.subspace.filter_param(), self.breakdown)
        # print("Query:" + sql)
        # res = data_source.excute_sql(sql)
        # print(res)
        # # x = []
        # # y = []
        # # if breakdown on the month
        # if self.breakdown == "arrival_date_month":
        #     res.sort(key=functools.cmp_to_key(self.sort_month))
        #
        # for item in res:
        #     self.result_x.append(item[self.breakdown])
        #     self.result_y.append(item[self.measure])
        global sql_call_count
        sql_call_count = sql_call_count + 1
        # try:
        self.result_x, self.result_y = execute_sql(self.measure, self.subspace.filter_param(), self.breakdown)
        # except Exception as e:
        #     print(e)
        #     print(self.measure, self.subspace.filter_dict, self.breakdown)
        #     sys.exit()
        # =======
        #         self.result_x, self.result_y = execute_sql(self.breakdown, self.measure, self.subspace.filter_param())
        #         self.result_y = list(map(lambda x: convert_digital_num(str(x)), self.result_y))
        self.smooth_result_y = pd.DataFrame({'Smooth': self.result_y}).ewm(alpha=0.8).mean()['Smooth'].values.tolist()
        # >>>>>>> 157e0d5a8a576b3a8f3cef44a544e277bc14eec1
        # print(f"======result_x:{self.result_x}")
        # print(f"======result_y:{self.result_y}")
        # print(f"======smooth_result_y:{self.smooth_result_y}")

        # return res

    def plot(self, insight=None):
        plt.plot(self.result_x, self.result_y, label=str(self.subspace.get_legend()))
        plt.scatter(self.result_x, self.result_y)
        plt.xlabel(self.breakdown)
        plt.ylabel(self.measure)
        plt.xticks(rotation=-20)
        plt.annotate(str({key: self.pattern_list[key] for key in insight}), (self.result_x[0], self.result_y[0]))
        # plt.annotate(str(self.pattern_list), (self.result_x[0], self.result_y[0]))

    def data_pattern_evaluate(self):
        # if not self.result_y:
        #     self.execute_data_source()

        if not self.pattern_list:
            global pattern_call_count
            pattern_call_count = pattern_call_count + 1
            self.pattern_list = evaluate_pattern(self.measure, self.subspace.filter_param(), self.breakdown)

        return self.pattern_list


class HomogenousDataScope:
    def __init__(self, initial_data_scope: DataScope, extend: str = None):
        """
        initial function of Homogenous Datasource
        :param initial_data_scope: initial expended data scope on which hds is generated
        :param extend: specify how to extend the initial data scope(specially, subspace:"subspace@column")
        """
        self.initial_data_scope = initial_data_scope
        if extend is None:
            self.extend = f"subspace@{initial_data_scope.subspace.get_columns()[0]}"
        else:
            self.extend = extend
        self.data_scope_list = []
        self.extend_dict = {"subspace": self.extend_by_subspace, "breakdown": self.extend_by_breakdown,
                            "measure": self.extend_by_measure}
        self.insight = []
        # self.common = []
        # self.exception = []
        self.common_exception_insight = {}
        self.impact = 0

    def extend_by_subspace(self):
        """
        extend the data scope by subspace
        :return:
        """
        extend_col = self.extend.split("@")[1]
        for distinct_value in distinct_dict[extend_col]:
            if distinct_value is not None:
                self.data_scope_list.append(
                    DataScope(self.initial_data_scope.subspace.extend(extend_col, distinct_value),
                              self.initial_data_scope.breakdown, self.initial_data_scope.measure))

    def extend_by_breakdown(self):
        """
        extend the data scope by breakdown
        :return:
        """
        pass

    def extend_by_measure(self):
        """
        extend the data scope by measure
        (extend space is so large that the result maybe not sensible enough)
        :return:
        """
        for measure in measure_list:
            self.data_scope_list.append(DataScope(
                self.initial_data_scope.subspace,
                self.initial_data_scope.breakdown, measure))

    def create_homogenous_data_scope(self):
        """
        create homogenous data scope
        :return:
        """
        self.extend_dict.get(self.extend.split("@")[0])()

    def execute_data_source(self):
        """
        execute the data scope from hds list
        :return:
        """
        for ds in self.data_scope_list:
            ds.execute_data_source()

    def plot(self, savefig=None):
        # pattern = []
        for ds in self.data_scope_list:
            print(f"x:{ds.result_x}")
            print(f"y:{ds.result_y}")
            print(ds.pattern_list)
            ds.plot(self.insight)
            # pattern.append(str(ds.subspace.filter_dict) + " : " + str(ds.pattern_list))
        print("================================")
        plt.legend(loc="best")
        # plt.text(0, 0, "\n".join(pattern), bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def pattern(self):
        for ds in self.data_scope_list:
            ds.data_pattern_evaluate()

    def insight_evaluate(self):
        potential_insight = [key for key in self.data_scope_list[0].data_pattern_evaluate().keys()]
        for ds in self.data_scope_list[1:]:
            potential_insight = [key for key in potential_insight if key in ds.data_pattern_evaluate().keys()]
            if not potential_insight:
                return False
        self.insight = potential_insight
        return self.insight

    def get_impact(self):
        self.impact = 0
        for ds in self.data_scope_list:
            self.impact = self.impact + ds.subspace.get_impact()
        # print("impact:", end=' ')
        # print(self.impact)
        return self.impact

    def common_exception_evaluate(self):
        high_light = {}
        for sight in self.insight:
            for ds in self.data_scope_list:
                ds_data_pattern_result = ds.data_pattern_evaluate()
                if type(ds_data_pattern_result[sight]) == list:
                    for it in ds_data_pattern_result[sight]:
                        t = (sight, it)
                        if t in high_light.keys():
                            high_light[t] = high_light[t] + 1
                        else:
                            high_light[t] = 1
                else:
                    t = (sight, ds_data_pattern_result[sight])
                    if t in high_light.keys():
                        high_light[t] = high_light[t] + 1
                    else:
                        high_light[t] = 1
        tao = 0.5
        score_dict = {}
        hds_sight = {"common_set": [], "exception": []}
        total = 0
        for item in high_light.items():
            total = total + item[1]
        for item in high_light.items():
            if item[1] / total > tao:
                hds_sight["common_set"].append(item[0])
            else:
                hds_sight["exception"].append(item[0])
        # print(hds_sight)
        self.common_exception_insight = hds_sight
        if len(self.common_exception_insight['common_set']) > 0:
            keys = self.common_exception_insight['common_set']
            y_list = [high_light[key] for key in keys]
            # print(self.common_exception_insight[sight]['common_set'])
            p_list = [i / total for i in y_list]
            # q_list = [1 / len(high_light) for l in y_list]
            cross_entropy_common_set = cross_entropy(p_list)
        else:
            cross_entropy_common_set = 0
        if len(self.common_exception_insight['exception']) > 0:
            keys = self.common_exception_insight['exception']
            y_list = [high_light[key] for key in keys]
            # print(self.common_exception_insight[sight]['exception'])
            p_list = [i / total for i in y_list]
            # q_list = [1 / len(high_light) for l in y_list]
            cross_entropy_exception = cross_entropy(p_list)
        else:
            cross_entropy_exception = 0
        r = 1
        s = cross_entropy_common_set + r * cross_entropy_exception
        k = 3
        if k < (1 - tao) * np.exp(1) / np.power(tao, 1 / r):
            s_max = -1 * np.log(tao) + r * k * np.power(tao, 1 / r) * np.log(np.exp(1)) / np.exp(1)
        else:
            s_max = -1 * tao * np.log(tao) - r * (1 - tao) * np.log((1 - tao) / k)
        # print(s_max)
        # print(f"=====s:{s}")
        gamma = 1.0

        if len(self.common_exception_insight['exception']) == 0:
            # print("no exception")
            conciseness = (s + gamma)
        else:
            conciseness = s
        # print(f"=====concise:{conciseness}")
        # conciseness = max(0, conciseness)
        impact_hds = self.get_impact()
        score = impact_hds * conciseness
        return score

