import os
import sys
import pandas as pd
from Prep import features_utills
import importlib
import time
from Couples import *
from modAL import ActiveLearner
import numpy as np
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import json
import csv
from tqdm import tqdm

importlib.reload(features_utills)
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

query_strategies = [
    # {"name": "entropy_sampling", "color": "#234f1d"},
    # {"name": "margin_sampling", "color": "#0000ff"},
    {"name": "uncertainty_sampling", "color": "#ff00d8"}
]

classifiers = [
    # {"name": "naive_bayes"},
    # {"name": "svm"},
    {"name": "random_forest"}
]


datasets = {#"abtBuy" : [[1, 5, 7, 8, 9, 10, 11], [7, 8, 9, 10], [14]],
            "abtBuy": [[2, 3, 7, 8, 9, 10, 12, 15], [9, 15], [14]],
            "DblpAcm": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
            "scholarDblp": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
            "amazonGoogleProducts": [[7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [14]]
            #'amazonGoogleProducts' : [[7, 8, 9, 10], [7, 8, 9, 10], [7, 8, 9, 10], [14]]
            }
configurations = [#"1block",
                  #"1al",
                  "05block_05al",
                  #"07block_03Al"#,
                  #"1block_1al"
                ]
# Configuration changes how the classifier will act, if just block then will use a Random Forest class, else will
# use an Active Learner classifier.
# conf = "07block_03Al"
# conf = "1block"
#conf = "1al"
# conf = "05block_05al" #half of the data used in the blocker + half data don't used
# conf = "1block_1al" #all the data used in the blocker as initial and then the same number as active learner
size_initial = 10
q_num = 1
results = {}


def saveResults(outName, results):
    data = json.dumps(results)
    out = open(outName + ".txt", "wt")
    out.write(data)
    out.close()
    pass


def initialPoolCreator(X_pool, y_pool, n_init):
    # Half of the initial training set is True and half False
    X_true = X_pool.loc[y_pool == 1]
    X_false = X_pool.loc[y_pool == 0]
    y_true = y_pool.loc[y_pool == 1]
    y_false = y_pool.loc[y_pool == 0]
    true_init = int(n_init / 2)
    false_init = int(n_init / 2)
    if n_init % 2 == 1:
        false_init = int(n_init / 2) + 1

    rand_true_list = [random.randint(0, len(X_true) - 1) for _ in range(true_init)]
    rand_false_list = [random.randint(0, len(X_false) - 1) for _ in range(false_init)]
    X_init = pd.concat([X_true.iloc[rand_true_list], X_false.iloc[rand_false_list]], axis=0)
    y_init = pd.concat([y_true.iloc[rand_true_list], y_false.iloc[rand_false_list]], axis=0)
    X_pool.drop(index=X_init.index, inplace=True)
    y_pool.drop(index=y_init.index, inplace=True)
    return X_init, X_pool, y_init, y_pool


if __name__ == '__main__':
    for conf in tqdm(configurations):
        for dataset, sim_config in datasets.items():
            results[dataset] = []
            path_data = os.path.join('..', 'Datasets', dataset)
            if dataset != "amazonGoogleProducts":
                dataset_left_name = os.path.join(path_data, 'dataset1.json')
                dataset_right_name = os.path.join(path_data, 'dataset2.json')
                dataset_gt_name = os.path.join(path_data, 'groundtruth.json')
                df1 = pd.read_json(dataset_left_name, lines=True)
                df1['realProfileID'] = df1['realProfileID'].astype("str")
                df1.set_index('realProfileID', inplace=True)
                df2 = pd.read_json(dataset_right_name, lines=True)
                df2['realProfileID'] = df2['realProfileID'].astype("str")
                df2.set_index('realProfileID', inplace=True)
                gt = pd.read_json(dataset_gt_name, lines=True)
                gt.rename(columns={'id1': 'id_l', 'id2': 'id_r'}, inplace=True)
                gt['id_l'] = gt['id_l'].astype("str")
                gt['id_r'] = gt['id_r'].astype("str")
            else:
                df1 = pd.read_csv(os.path.join(path_data, 'Amazon.csv'), encoding='latin-1')
                df1.rename(columns={'title': 'name'}, inplace=True)
                df1.set_index('id', inplace=True)
                df2 = pd.read_csv(os.path.join(path_data, 'GoogleProducts.csv'), encoding='latin-1')
                df2.set_index('id', inplace=True)
                gt = pd.read_csv(os.path.join(path_data, 'Amzon_GoogleProducts_perfectMapping.csv'), encoding='latin-1')
                # id_l == idAmazon, id_r == url google
                gt.columns = ['id_l', 'id_r']

            blk = pd.read_parquet("../" + "Couples/" + dataset)

            for budget_train in range(100, 600, 100):

                raw_used_couples = pd.read_csv(
                    f"../blocking_al/already_labeled/already_labeled_{dataset}_{budget_train}_{classifiers[0]['name']}_{query_strategies[0]['name']}.csv")
                raw_couples = pd.read_csv(
                    f"../blocking_al/possible_couples/possible_couples_{dataset}_{budget_train}_{classifiers[0]['name']}_{query_strategies[0]['name']}.csv")
                raw_couples.rename(columns={"Unnamed: 0": "pair_id"}, inplace=True)
                raw_used_couples.rename(columns={"Unnamed: 0": "pair_id"}, inplace=True)
                raw_used_couples.head()

                # find the pairs with respective ids of the couple
                couples = blk.loc[raw_couples['pair_id']]
                used_couples = blk.loc[raw_used_couples['pair_id']]
                # Get from files the pairs from the blocker and the pairs used in the blocking active learner
                pairs = []
                used_pairs = []
                for index, row in couples.iterrows():
                    pairs.append((row["profileID1"], row["profileID2"]))
                    try:
                        df1.loc[row["profileID1"]]
                    except:
                        print(f"error in {row['profileID1']} and {row['profileID2']}")
                for index, row in used_couples.iterrows():
                    used_pairs.append((row["profileID1"], row["profileID2"]))
                    try:
                        df1.loc[row["profileID1"]]
                    except:
                        print(f"error in {row['profileID1']} and {row['profileID2']}")
                pairs = set(pairs)
                used_pairs = set(used_pairs)
                # Use the already calculated df_features from existing files
                sim_config_string = ""
                for v in sim_config:
                    sim_config_string += f"{str(v)}_"
                file_df_features_name = f'df_features/df_features_{dataset}_{sim_config_string}{budget_train}.csv'
                file_df_used_features_name = f'df_features/df_used_features_{dataset}_{sim_config_string}{budget_train}.csv'
                # Trow error if don't find file
                df_features = features_utills.getDf_featuresFromFile(file_df_features_name, 'id_l', 'id_r')
                df_used_features = features_utills.getDf_featuresFromFile(file_df_used_features_name, 'id_l', 'id_r')

                col_X = []
                for i in df_features.columns:
                    if i != 'label' and i != 'id_r' and i != 'id_l':
                        col_X.append(i)

                X = df_features.loc[:, col_X]
                y = df_features['label']
                X_used = df_used_features.loc[:, col_X]
                y_used = df_used_features['label']
                print(f"used_pairs {len(used_pairs)} ")
                print(f" pairs {len(pairs)} ")
                # Classifier get only the already labeled data used in the blocking
                if conf == "1block":
                    clf = RandomForestClassifier(n_estimators=100)
                    clf.fit(X_used, y_used)
                    y_pred_forest = clf.predict(X)
                    results[dataset].append(metrics.recall_score(y, y_pred_forest))
                    results[dataset].append(metrics.precision_score(y, y_pred_forest))
                    results[dataset].append(metrics.f1_score(y, y_pred_forest))

                # Classifier use only active learning with the pairs not used in the blocker
                elif conf == "1al":
                    X_init, X_pool, y_init, y_pool = initialPoolCreator(X, y, size_initial)
                    learner = ActiveLearner(
                        estimator=RandomForestClassifier(n_estimators=100),
                        X_training=X_init,
                        y_training=np.ravel(y_init)
                    )
                    # Remember indexes of learner's choices
                    learner_choices = []
                    for n in range(int((budget_train - size_initial) / q_num)):
                        query_idx, query_inst = learner.query(X_pool, n_instances=q_num)
                        y_new = y_pool.loc[query_inst.index]
                        try:
                            learner.teach(query_inst,np.ravel(y_new))
                            learner_choices.append(query_inst.index.tolist())
                        except:
                            print(query_inst.index, y_new)
                        learner_choices.append(query_inst.index.tolist())
                        X_pool = X_pool.drop(index=query_inst.index)
                        y_pool = y_pool.drop(index=query_inst.index)

                    y_pred_learner = learner.predict(X_pool)
                    results[dataset].append(metrics.recall_score(y_pool, y_pred_learner))
                    results[dataset].append(metrics.precision_score(y_pool, y_pred_learner))
                    results[dataset].append(metrics.f1_score(y_pool, y_pred_learner))
                    # Learner choices len is budget - n_initial
                    saveResults(f"learner_choices_{dataset}_{budget_train}", learner_choices)


                elif conf == "05block_05al":
                    X_incomplete_init, X_pool, y_incomplete_init, y_pool = initialPoolCreator(X, y,
                                                                                              int(size_initial / 2))
                    X_used_incomplete_init, X_used_pool, y_used_incomplete_init, y_used_pool = initialPoolCreator(
                        X_used,
                        y_used,
                        int(
                            size_initial / 2))

                    X_init = pd.concat([X_incomplete_init, X_used_incomplete_init], axis=0)
                    y_init = pd.concat([y_incomplete_init, y_used_incomplete_init], axis=0)

                    learner = ActiveLearner(
                        estimator=RandomForestClassifier(n_estimators=100),
                        X_training=X_init,
                        y_training=np.ravel(y_init),
                    )
                    # MUST CONTINUE HERE
                    # Half of queries are from the data don't used in the blocking
                    # Half of queries are from the data don't used in the blocking
                    for n in range(int(budget_train - size_initial / 2 * q_num)):
                        _, query_inst = learner.query(X_pool, n_instances=q_num)
                        y_new = y_pool.loc[query_inst.index]
                        learner.teach(query_inst, np.ravel(y_new))
                        X_pool = X_pool.drop(index=query_inst.index)
                        y_pool = y_pool.drop(index=query_inst.index)
                        _, query_inst = learner.query(X_used_pool, n_instances=q_num)
                        y_new = y_used_pool.loc[query_inst.index]
                        learner.teach(query_inst, np.ravel(y_new))
                        X_used_pool = X_used_pool.drop(index=query_inst.index)
                        y_used_pool = y_used_pool.drop(index=query_inst.index)

                    y_pred_learner = learner.predict(pd.concat([X_pool, X_used_pool], axis=0))
                    results[dataset].append(
                        metrics.recall_score(pd.concat([y_pool, y_used_pool], axis=0), y_pred_learner))
                    results[dataset].append(
                        metrics.precision_score(pd.concat([y_pool, y_used_pool], axis=0), y_pred_learner))
                    results[dataset].append(
                        metrics.f1_score(pd.concat([y_pool, y_used_pool], axis=0), y_pred_learner))  #
                #Use all the data labeled from blocking and budget/2 classifier choices
                elif conf == "07block_03Al":

                    learner = ActiveLearner(
                        estimator=RandomForestClassifier(n_estimators=100),
                        X_training=X_used,
                        y_training=np.ravel(y_used),
                    )
                    X_pool = X.copy()
                    y_pool = y.copy()
                    q_num = 1
                    for n in range(int(budget_train / 2 * q_num)):
                        _, query_inst = learner.query(X_pool, n_instances=q_num)
                        y_new = y_pool.loc[query_inst.index]
                        learner.teach(query_inst, np.ravel(y_new))
                        X_pool = X_pool.drop(index=query_inst.index)
                        y_pool = y_pool.drop(index=query_inst.index)

                    y_pred_learner = learner.predict(X_pool)
                    results[dataset].append(metrics.recall_score(y_pool, y_pred_learner))
                    results[dataset].append(metrics.precision_score(y_pool, y_pred_learner))
                    results[dataset].append(metrics.f1_score(y_pool, y_pred_learner))

                elif conf == "1block_1al":

                    learner = ActiveLearner(
                        estimator=RandomForestClassifier(n_estimators=100),
                        X_training=X_used,
                        y_training=np.ravel(y_used),
                    )
                    X_pool = X.copy()
                    y_pool = y.copy()
                    q_num = 1
                    for n in range(int(budget_train / q_num)):
                        query_idx, query_inst = learner.query(X_pool, n_instances=q_num)
                        y_new = y_pool.loc[query_inst.index]
                        learner.teach(query_inst, np.ravel(y_new))
                        X_pool = X_pool.drop(index=query_inst.index)
                        y_pool = y_pool.drop(index=query_inst.index)

                    y_pred_learner = learner.predict(X_pool)
                    results[dataset].append(metrics.recall_score(y_pool, y_pred_learner))
                    results[dataset].append(metrics.precision_score(y_pool, y_pred_learner))
                    results[dataset].append(metrics.f1_score(y_pool, y_pred_learner))

        with open(f"{conf}_metricsWithBlockerAl.csv", "w") as file:
            writ = csv.writer(file)
            fieldnames = ["dataset", "rec_100", "pre_100", "f1_100", "rec_200", "pre_200", "f1_200", "rec_300",
                          "pre_300",
                          "f1_300", "rec_400", "pre_400", "f1_400", "rec_500", "pre_500", "f1_500"]

            col = results.keys()
            writ.writerow(fieldnames)
            for key, value in results.items():
                value.insert(0, key)
                writ.writerow(value[n] for n in range(len(value)))

""" import json 
    for data in datasets:
        with open(os.path.join("..","blocking_al",f"{data}_C12_run0.txt"),"r") as json_file:
            out=json.load(json_file)
            csv_file=open(os.path.join("..","blocking_al",f"{data}_C12_run0.csv"),"w")
            writer = csv.writer(csv_file)
            fieldnames=["budget","random_recall","random_precision","random_f1","uncertainty_sampling_recall","uncertainty_sampling_precision","uncertainty_sampling_f1"]
            writer.writerow(fieldnames)
            
            for n in out['random_forest']:
                #clear_list=list(out['random_forest'][n["budget"]]['results'].values())
                clear_list=list(n["results"].values())[0]
                clear_list+=(list(n["results"].values())[1])
                clear_list.insert(0,n["budget"])
                writer.writerow(clear_list)
            csv_file.close()
            """
