import json
import csv
import queue
from config import *
from data_source import *
from metainsight.insight import *
from superinsight.feature_vector import *
import yaml
from sklearn.preprocessing import StandardScaler
import joblib
from models import knn
from models import linear_regression
from models import random_forest
from models import xgboost_model
from models import lightgbm_model
from models import catboost_model
from models import neural_network
import torch

global dataset
dataset = ""
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


def init_dataset(num=1, dataset="hotel_booking"):
    db_pool = DataSource(host=host, user=user, password=password, database=database, num_thread=num)
    insight_data_source(db_pool, num)
    feature_data_source(db_pool)

    # test case for data scope & hds
    # dataset = "hotel_booking"
    # set_dataset(dataset)

    # initialize the information of dataset for later usage
    get_record_num(dataset)
    get_distinct_value(dataset)
    measure_list_generate(dataset)


def get_column_meta_data(dataset):
    with open(f"dataset_example/config/{dataset}.json", 'r', encoding='utf8') as fp:
        json_data = json.load(
            fp)  # {'Year': 'Temporal', 'Brand': 'Categorical', 'Category': 'Categorical', 'Model': 'Categorical', 'Sales': 'Numerical'}
        global column_meta_data
        column_meta_data = {
            dataset: json_data}  # {'CarSales': {'Year': 'Temporal', 'Brand': 'Categorical', 'Category': 'Categorical', 'Model': 'Categorical', 'Sales': 'Numerical'}}
        return column_meta_data


def create_data_scope(dataset, pipeline=False):
    # todo: try to calculate the impact of each subspace first
    #  and then generate all possible triple tuple
    fifo_ds = []
    count_prune_ds = []
    if pipeline == False:
        f1 = open(f'dataset_example/test_{dataset}_fifo.csv', 'w', encoding='utf-8')
        csv_writer1 = csv.writer(f1)
        csv_writer1.writerow(["subspace", "breakdown", "measure"])
        f2 = open(f'dataset_example/test_{dataset}_count_prune.csv', 'w', encoding='utf-8')
        csv_writer2 = csv.writer(f2)
        csv_writer2.writerow(["subspace", "breakdown", "measure"])
    data_scope_queue = queue.PriorityQueue()
    for subspace_col, subspace_col_type in column_meta_data[dataset].items():
        if subspace_col_type != "Numerical":
            for distinct_value in distinct_dict[subspace_col]:
                subspace = Subspace({subspace_col: distinct_value})
                # subspace.get_impact()
                for breakdown_col, breakdown_col_type in column_meta_data[dataset].items():
                    if breakdown_col_type != "Numerical" and subspace_col != breakdown_col:
                        for measure_col, measure_col_type in column_meta_data[dataset].items():
                            if measure_col_type == "Numerical":
                                # for agg in aggregate_function:
                                agg = "MAX"
                                measure = "{}({})".format(agg, measure_col)
                                data_scope = DataScope(subspace, breakdown_col, measure)
                                data_scope_queue.put(data_scope)
                                if pipeline == False:
                                    csv_writer1.writerow([subspace.filter_param(), breakdown_col, measure_col])
                                else:
                                    fifo_ds.append([subspace.filter_param(), breakdown_col, measure_col])
                                print(data_scope_queue.qsize())
    while not data_scope_queue.empty():
        ds = data_scope_queue.get()
        # csv_writer.writerow([subspace.filter_param(), breakdown_col, measure])
        if pipeline == False:
            csv_writer2.writerow([ds.subspace.filter_param(), ds.breakdown, ds.measure[4:-1]])
        else:
            count_prune_ds.append([ds.subspace.filter_param(), ds.breakdown, ds.measure[4:-1]])

    f1.close()
    f2.close()
    if pipeline == False:
        return data_scope_queue
    else:
        return fifo_ds, count_prune_ds


hds_exitFlag = 0
lock = threading.Lock()
global hds_cur
hds_finished = False


def calculate_hds_worker(work_queue: queue, writer):
    global hds_cur
    global hds_exitFlag
    global hds_finished
    start_time = time.perf_counter()
    while not hds_exitFlag:
        lock.acquire()
        if not work_queue.empty():
            hds_cur = hds_cur + 1

            row = work_queue.get()
            lock.release()
            ds_kv = row[0].split(" = ")
            ds_dict = {ds_kv[0]: ds_kv[1].replace("\"", "")}
            subspace = Subspace(ds_dict)
            for agg in aggregate_function:
                measure = "{}({})".format(agg, row[2])
                ds = DataScope(subspace, row[1], measure)
                if ds.data_pattern_evaluate():
                    hds = HomogenousDataScope(ds)
                    # hds = HomogenousDataScope(data_scope,"measure")
                    hds.create_homogenous_data_scope()
                    # execute & plot
                    hds.execute_data_source()
                    if hds.insight_evaluate():
                        hds.common_exception_evaluate()
                        score = hds.common_exception_evaluate()
                        end_time = time.perf_counter()
                        print(f"\r Current : {hds_cur}    Rest:{work_queue.qsize()}     {end_time - start_time}",
                              end='',
                              flush=True)
                        writer.writerow([ds.subspace.filter_param(), ds.breakdown, ds.measure,
                                         end_time - start_time, score])

                        if hds_finished and work_queue.empty():
                            hds_exitFlag = 1
                            print("done!")
                    else:
                        end_time = time.perf_counter()
                        score = 10
                        writer.writerow([ds.subspace.filter_param(), ds.breakdown, ds.measure,
                                         end_time - start_time, score])
                        print(f"\r Current : {hds_cur}    Rest:{work_queue.qsize()}     {end_time - start_time}",
                        end='', flush=True)
                        if hds_finished and work_queue.empty():
                            hds_exitFlag = 1
        else:
            lock.release()


def calculate_corpus_worker(work_queue: queue, writer):
    global hds_cur
    global hds_exitFlag
    global hds_finished
    start_time = time.perf_counter()

    while not hds_exitFlag:
        lock.acquire()
        if not work_queue.empty():
            # count = count + 1
            hds_cur = hds_cur + 1
            initial_score = 10
            row = work_queue.get()
            lock.release()
            # print(row)
            if " = " in row[0]:
                ds_kv = row[0].split(" = ")
                ds_dict = {ds_kv[0]: ds_kv[1].replace("\"", "")}
                subspace = Subspace(ds_dict)
                for agg in aggregate_function:
                    measure = "{}({})".format(agg, row[2])
                    ds = DataScope(subspace, row[1], measure)
                    if ds.data_pattern_evaluate():
                        hds = HomogenousDataScope(ds)
                        # hds = HomogenousDataScope(data_scope,"measure")
                        hds.create_homogenous_data_scope()
                        # execute & plot
                        hds.execute_data_source()
                        if hds.insight_evaluate():
                            hds.common_exception_evaluate()

                            score = hds.common_exception_evaluate()
                            if score < initial_score:
                                initial_score = score
                writer.writerow([ds.subspace.filter_param(), ds.breakdown, row[2], initial_score])
            end_time = time.perf_counter()
            # print([ds.subspace.filter_param(), ds.breakdown, row[2], score])
            print(f"\r Current : {hds_cur}    Rest:{work_queue.qsize()}     {end_time - start_time}",
                  end='', flush=True)
        else:
            lock.release()


def generate_corpus(dataset, sort_type, modelname=None, pipeline=False):
    if modelname != None:
        calculate_hds_from_csv(f'dataset_example/sorted_{dataset}_{sort_type}_score_{modelname}.csv',
                               f'dataset_example/res_{dataset}_{sort_type}_score_{modelname}.csv', 8)
    else:
        calculate_hds_from_csv(f'dataset_example/test_{dataset}_{sort_type}.csv',
                               f'dataset_example/res_{dataset}_{sort_type}.csv', 8)


def calculate_hds_from_csv(ds_file, res_file, thread_num=1, calculate=True):
    f_r = open(ds_file, 'r', encoding='utf-8')
    csv_reader = csv.reader(f_r)
    headers = next(csv_reader)

    f_w = open(res_file, 'w', encoding='utf-8')
    csv_writer = csv.writer(f_w)

    # create thread

    threads = []
    work_queue = queue.Queue()

    global count
    global hds_cur
    for i in range(thread_num):
        if calculate:
            t = threading.Thread(target=calculate_hds_worker, args=(work_queue, csv_writer))
        else:
            t = threading.Thread(target=calculate_corpus_worker, args=(work_queue, csv_writer))
        threads.append(t)
        t.start()
    count = 0
    hds_cur = 0
    for row in csv_reader:
        if row:
            lock.acquire()
            work_queue.put(row)
            lock.release()
            count += 1
    global hds_finished
    hds_finished = True

    for t in threads:
        t.join()

    f_r.close()
    f_w.close()


def calculate_feature_vector_2(dataset, sort_type, pipeline=False, ds_list=None):
    exe_count = 0
    data_list = []
    if pipeline == False:
        f_r = open(f'dataset_example/test_{dataset}_{sort_type}.csv', "r", encoding='utf-8')
        csv_reader = csv.reader(f_r)
        headers = next(csv_reader)
    else:
        csv_reader = ds_list
    for row in csv_reader:
        if row:
            if " = " in row[0]:
                ds_kv = row[0].split(" = ")
                ds_dict = {ds_kv[0]: ds_kv[1].replace("\"", "")}
                subspace = Subspace(ds_dict).filter_param()
                subspace_feature = calculate_feature_with_category(ds_kv[0])
                breakdown_feature = calculate_feature_with_category(row[1], subspace)
                measure_feature = calculate_feature_with_category(row[2], subspace)
                feature_vector = np.append(subspace_feature, breakdown_feature)
                feature_vector = np.append(feature_vector, measure_feature)
                data_list.append(feature_vector)
                exe_count = exe_count + 1
                print(f"\r Current : {exe_count}", end='',
                      flush=True)
            else:
                exe_count = exe_count + 1
                print(f"\r Current : {exe_count}", end='',
                      flush=True)
    f_r.close()
    if pipeline == False:
        f_r = open(f'dataset_example/test_{dataset}_{sort_type}.csv', "r", encoding='utf-8')
        csv_reader = csv.reader(f_r)
        headers = next(csv_reader)
        exe_count = 0
        f_w = open(f'dataset_example/test_featured_{dataset}_{sort_type}.csv', "w", encoding='utf-8')
        csv_writer = csv.writer(f_w)
        for row in csv_reader:
            if row:
                if " = " in row[0]:
                    csv_writer.writerow([row[0], row[1], row[2], row[3], data_list[exe_count]])
                    exe_count = exe_count + 1
        f_r.close()
        f_w.close()
    else:
        return data_list


def calculate_score_by_model_from_dataset(dataset, sort_type, modelname, pipeline=False, ds_list=None):
    data_list = []
    exe_count = 0
    if pipeline == False:
        f_r = open(f"dataset_example/test_{dataset}_{sort_type}.csv", "r", encoding='utf-8')
        csv_reader = csv.reader(f_r)
        headers = next(csv_reader)
    else:
        csv_reader = ds_list
    for row in csv_reader:
        if row:
            ds_kv = row[0].split(" = ")
            ds_dict = {ds_kv[0]: ds_kv[1].replace("\"", "")}
            subspace = Subspace(ds_dict).filter_param()
            subspace_feature = calculate_feature(ds_kv[0])
            breakdown_feature = calculate_feature(row[1], subspace)
            measure_feature = calculate_feature(row[2][4:-1], subspace)
            feature_vector = np.append(subspace_feature, breakdown_feature)
            feature_vector = np.append(feature_vector, measure_feature)
            data_list.append(feature_vector)
            exe_count = exe_count + 1
            print(f"\r Current : {exe_count}", end='',
                  flush=True)
    f_r.close()

    scale = StandardScaler()
    data_list = scale.fit_transform(data_list)
    model_lr = joblib.load('model/linear_regression_model_with_no_insight_score_with_standard_scale.m')
    model_knn = joblib.load('model/knn_model_with_no_insight_score_with_standard_scale.m')
    model_rf = joblib.load('model/random_forest_model_with_no_insight_score_with_standard_scale.m')
    model_xgb = joblib.load('model/xgboost_model_with_no_insight_score_with_standard_scale.m')
    model_lgb = joblib.load('model/lightgbm_model_with_no_insight_score_with_standard_scale.m')
    model_cat = joblib.load('model/catboost_model_with_no_insight_score_with_standard_scale.m')
    model_mlp = torch.load('model/neural_network_model_with_no_insight_score.m')
    data_list = np.array(data_list).astype('float64')
    score_lr = linear_regression.predict(model_lr, data_list)
    score_knn = knn.predict(model_knn, data_list)
    score_rf = random_forest.predict(model_rf, data_list)
    score_xgb = xgboost_model.predict(model_xgb, data_list)
    score_lgb = lightgbm_model.predict(model_lgb, data_list)
    score_cat = catboost_model.predict(model_cat, data_list)
    score_mlp = neural_network.predict(model_mlp, data_list)
    if pipeline == False:
        f_r = open(f"dataset_example/test_{dataset}_{sort_type}.csv", "r", encoding='utf-8')
        csv_reader = csv.reader(f_r)
        headers = next(csv_reader)
        f_w_knn = open(f"dataset_example/test_{dataset}_{sort_type}_score_knn.csv", "a", encoding='utf-8')
        f_w_lr = open(f"dataset_example/test_{dataset}_{sort_type}_score_linear_regression.csv", "a", encoding='utf-8')
        f_w_rf = open(f"dataset_example/test_{dataset}_{sort_type}_score_random_forest.csv", "a", encoding='utf-8')
        f_w_mlp = open(f"dataset_example/test_{dataset}_{sort_type}_score_mlpnn.csv", "w", encoding='utf-8')
        f_w_xgb = open(f"dataset_example/test_{dataset}_{sort_type}_score_xgboost.csv", "w", encoding='utf-8')
        f_w_lgb = open(f"dataset_example/test_{dataset}_{sort_type}_score_lightgbm.csv", "w", encoding='utf-8')
        f_w_cat = open(f"dataset_example/test_{dataset}_{sort_type}_score_catboost.csv", "w", encoding='utf-8')
        csv_writer_knn = csv.writer(f_w_knn)
        csv_writer_lr = csv.writer(f_w_lr)
        csv_writer_rf = csv.writer(f_w_rf)
        csv_writer_xgb = csv.writer(f_w_xgb)
        csv_writer_lgb = csv.writer(f_w_lgb)
        csv_writer_cat = csv.writer(f_w_cat)
        csv_writer_mlp = csv.writer(f_w_mlp)
        exe_count = 0
        for row in csv_reader:
            if row:
                csv_writer_knn.writerow([row[0], row[1], row[2], score_knn[exe_count]])
                csv_writer_lr.writerow([row[0], row[1], row[2], score_lr[exe_count]])
                csv_writer_rf.writerow([row[0], row[1], row[2], score_rf[exe_count]])
                csv_writer_xgb.writerow([row[0], row[1], row[2], score_xgb[exe_count]])
                csv_writer_lgb.writerow([row[0], row[1], row[2], score_lgb[exe_count]])
                csv_writer_cat.writerow([row[0], row[1], row[2], score_cat[exe_count]])
                csv_writer_mlp.writerow([row[0], row[1], row[2], score_mlp[exe_count]])
                exe_count = exe_count + 1
                print(f"\r Current : {exe_count}", end='',
                      flush=True)

            # ds = DataScope(subspace, row[1], row[2])
        f_r.close()
        f_w_knn.close()
        f_w_lr.close()
        f_w_rf.close()
        f_w_xgb.close()
        f_w_lgb.close()
        f_w_cat.close()
        f_w_mlp.close()
    else:
        if modelname == "knn":
            return score_knn
        elif modelname == "linear_regression":
            return score_lr
        elif modelname == "random_forest":
            return score_rf
        elif modelname == "mlpnn":
            return score_mlp
        elif modelname == "lightgbm":
            return score_lgb
        elif modelname == "xgboost":
            return score_xgb
        else:
            return score_cat


def sort_score_from_csv(dataset, sort_type, modelname):
    df = pd.read_csv(f'dataset_example/test_{dataset}_{sort_type}_score_{modelname}.csv',
                names=['1', '2', '3', '4'])
    new_df = df.sort_values(by="4", ascending=True)
    new_df.to_csv(f'dataset_example/sorted_{dataset}_{sort_type}_score_{modelname}.csv', index=False)


def calculate(dataset, sort_type, modelname=None):
    time_line = []
    score_list = []
    insight_list = []
    cur_insight = []
    k = 5
    count_insight = {}
    if modelname == None:
        filename = f'dataset_example/res_{dataset}_{sort_type}.csv'
    else:
        filename = f'dataset_example/res_{dataset}_{sort_type}_score_{modelname}.csv'
    with open(filename, "r") as r:
        csv_reader = csv.reader(r)
        time_budget = 10
        score_sum = 0.0

        for row in csv_reader:
            if row:
                if float(row[3]) > time_budget:
                    time_line.append(time_budget)
                    key_num = count_insight.__len__()
                    if insight_list.__len__() != 0:
                        # cur_sum = score_sum * key_num / insight_list.__len__() / 3
                        cur_sum = score_sum * key_num / 5 / 3
                    else:
                        cur_sum = 0
                    score_list.append(cur_sum)
                    time_budget = time_budget + 2
                    if time_budget > 300:
                        break
                score = 1.0 - float(row[4]) / 6
                score_sum = score_sum + score
                insight_list.append([row[0], row[1], row[2], score])
                if row[0] not in count_insight:
                    count_insight[row[0]] = 0
                if row[1] not in count_insight:
                    count_insight[row[1]] = 0
                if row[2] not in count_insight:
                    count_insight[row[2]] = 0
                count_insight[row[0]] = count_insight[row[0]] + 1
                count_insight[row[1]] = count_insight[row[1]] + 1
                count_insight[row[2]] = count_insight[row[2]] + 1

                key_num = count_insight.__len__()

                if insight_list.__len__() > k:
                    index = 0
                    maxValue = 0.0
                    for i in range(k + 1):
                        delete = 0
                        if count_insight[insight_list[i][0]] == 1:
                            delete = delete + 1
                        if count_insight[insight_list[i][1]] == 1:
                            delete = delete + 1
                        if count_insight[insight_list[i][2]] == 1:
                            delete = delete + 1
                        cur_sum = (score_sum - insight_list[i][3]) * (key_num - delete) / k / 3
                        if cur_sum > maxValue:
                            maxValue = cur_sum
                            index = i
                    score_sum = score_sum - insight_list[index][3]
                    count_insight[insight_list[index][0]] = count_insight[insight_list[index][0]] - 1
                    if count_insight[insight_list[index][0]] == 0:
                        count_insight.pop(insight_list[index][0])
                    count_insight[insight_list[index][1]] = count_insight[insight_list[index][1]] - 1
                    if count_insight[insight_list[index][1]] == 0:
                        count_insight.pop(insight_list[index][1])
                    count_insight[insight_list[index][2]] = count_insight[insight_list[index][2]] - 1
                    if count_insight[insight_list[index][2]] == 0:
                        count_insight.pop(insight_list[index][2])
                    insight_list.pop(index)
    global res_list
    res_list = insight_list
    return time_line, score_list


def recommend_top_k_insights(dataset, sort_type, modelname=None, pipeline=False):
    sort_map = {
        'fifo': "MetaInsight(FIFO)",
        'count_prune': "MetaInsight(Count_Prune)",
        'random_forest': "SuperInsight(RF)",
        'mlpnn': "SuperInsight(MLPNN)",
        'linear_regression': "SuperInsight(LR)"
    }
    res_list = []
    series = []
    cur = {}
    d = dataset
    s = sort_type
    cur['name'] = sort_map[s]
    cur['type'] = 'line'
    cur['symbolSize'] = 6
    time_line, score_list = calculate(d, s, modelname)
    cur['data'] = score_list
    series.append(cur)
    print(res_list)


def explore1(dataset, sort_type, modelname=None, pipeline=False):
    if pipeline == True:
        if modelname == None: # metainsight
            # get_default_config()
            # get_column_meta_data(dataset)
            # init_dataset(8)
            # fifo_ds, count_prune_ds = create_data_scope(dataset, pipeline=True)
            # if sort_type == "fifo":
            #     generate_corpus(dataset, sort_type)
            # else:
            pass
        else:
            pass
        pass
    else: # save result as csv-file
        if modelname == None:  # metainsight
            get_default_config()
            get_column_meta_data(dataset)
            init_dataset(8)
            create_data_scope(dataset)
            generate_corpus(dataset, sort_type,  pipeline=False)
            recommend_top_k_insights(dataset, sort_type)
        else:  # superinsight
            get_default_config()
            get_column_meta_data(dataset)
            init_dataset(8)
            fifo_ds, count_prune_ds = create_data_scope(dataset)
            calculate_feature_vector_2(dataset, sort_type)
            calculate_score_by_model_from_dataset(dataset, sort_type)
            sort_score_from_csv(dataset, sort_type, modelname)
            generate_corpus(dataset, sort_type, modelname=modelname, pipeline=False)
            recommend_top_k_insights(dataset, sort_type, modelname=modelname)


if __name__ == '__main__':
    explore1("CarSales", "fifo", modelname="random_forest", pipeline=False)
    dataset = "CarSales"