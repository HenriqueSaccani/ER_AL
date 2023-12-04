import os
import sys
import pandas as pd
from Prep import features_utills
import importlib
from Couples import *
from modAL import ActiveLearner
import numpy as np
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import json
import pickle
from tqdm import tqdm

importlib.reload(features_utills)
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
'''
This code uses the datasets created by "Notebooks/df_feature_creator.py" and uses active Learning, 
with a Random forest classifier, to calculate the metrics f1-score, precision and recall.

To teach the active learner 2 different types of pairs are used:
block data : this pairs are used during the blocking to teach another active learning, see "blocking_al/couples_creator"
             for more details.
al data : This pairs are choose by the active learning in THIS code.

This code use different percentages of this 2 types of data, never surpassing the 100 and 500 budget limit.
For example, when using 100 examples as budget, if in the blocking were used 40 samples, the active learner will
be taught with this 40 samples and will choose another 60 samples for the remaining pool.

The minimum value of the blocking data used is 10% of the total budget.
For each budget 10 test are calculated : 10block_90al, 20block_80al... 100block_0al ---- budget = 100
                                         50block_450al, 100block_400al... 500block_0al ---- budget = 500
                                         
The results are saved and are plotted in the "Notebooks/Plots/plots_block&al_standard_creator.py" code.
'''
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

'''
Used to find all the similarities from datasets,
'''
datasets = {
    "abtBuy": [[2, 3, 7, 8, 9, 10, 12, 15], [9, 15], [14]],
    "DblpAcm": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
    "scholarDblp": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
    "amazonGoogleProducts": [[7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [14]]
}

configurations = [##"block&al_standard",
                  # This configuration use an active learner based on RandomForest to choose samplings depending on
                  # the results of the similarity features
                  # "half_true&half_false"
                  # This configurat mxion use a RandomForestClassifier trained by half True labels and half False Labels
                  ##"block&al_standard_tesis" (NOT WORKING, MISSING POSSIBLE COUPLES)
                  #Same as standard but uses 100,200,300,400 and 500
                  ]

q_num = 1
results = {}
first_time = True  # For some reason python execute the code below the if this aux is used before executing what is inside the if


def saveResults(outName, results):
    data = json.dumps(results)
    out = open(outName + ".txt", "wt")
    out.write(data)
    out.close()
    pass


def getX_non_used_y_non_used_X_used_y_used(dataset, budget_train, classifier_name, query_strategy_name, blk,
                                           sim_config):
    raw_used_couples = pd.read_csv(
        f"../blocking_al/already_labeled/already_labeled_{dataset}_{budget_train}_{classifier_name}_{query_strategy_name}.csv")
    raw_couples = pd.read_csv(
        f"../blocking_al/possible_couples/possible_couples_{dataset}_{budget_train}_{classifier_name}_{query_strategy_name}.csv")
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

    # X_non used contain the data who hasn't been chosen by the blocker Al
    X_non_used = df_features.loc[:, col_X]
    y_non_used = df_features['label']
    # X_used contain the data that has been chosen by the blocker Al
    X_used = df_used_features.loc[:, col_X]
    y_used = df_used_features['label']
    return X_non_used, y_non_used, X_used, y_used


if __name__ == '__main__':
    for conf in tqdm(configurations):
        for dataset, sim_config in tqdm(datasets.items()):
            results[dataset] = {}
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

            if conf == "block&al_standard":
                for total_budget in range(100, 600, 400):  # total_budget 100 and 500
                #for total_budget in range(100, 101, 10):  # total_budget is 100
                    for block_percentage in range(10, 110, 10):  # if tot_budget = 100 start on 10 and end in 100
                        # initialization
                        learner = ActiveLearner(
                            estimator=RandomForestClassifier(n_estimators=100),
                        )
                        X_non_used, y_non_used, X_used, y_used = getX_non_used_y_non_used_X_used_y_used(
                            dataset,
                            int(block_percentage / 100 * total_budget),
                            classifiers[0]['name'],
                            query_strategies[0]['name'],
                            blk.copy(),
                            sim_config
                        )
                        al_percentage = 100 - block_percentage
                        # if one of the percentages is equal to 0 the function is a little bit different
                        if block_percentage == 100:
                            # use 100% from the blocker data
                            X_test = X_non_used.copy()
                            y_test = y_non_used.copy()
                            learner.teach(X_used, np.ravel(y_used))
                            # al choose from all the samples

                        else:

                            print(f"blk% {block_percentage}, al% {al_percentage}")
                            print(f"X used {X_used.shape}")
                            print(f"X non_used {X_non_used.shape}")

                            # Training the learner using the data already used by the Block Al
                            learner.teach(X_used, np.ravel(y_used))

                            X_al_pool = X_non_used.copy()
                            y_al_pool = y_non_used.copy()

                            # Training the learner letting he choose from all the pairs remaining
                            for n in range(int((total_budget * (al_percentage / 100)) / q_num)):
                                query_idx, query_inst = learner.query(X_al_pool, n_instances=q_num)
                                y_new = y_al_pool.loc[query_inst.index]
                                try:
                                    learner.teach(query_inst, np.ravel(y_new))
                                except:
                                    print("Error in al percentage for in index: ")
                                X_al_pool = X_al_pool.drop(index=query_inst.index)
                                y_al_pool = y_al_pool.drop(index=query_inst.index)

                            X_test = X_al_pool.copy()
                            y_test = y_al_pool.copy()

                        y_pred_learner = learner.predict(X_test)
                        percentageString = str(f'block{block_percentage}_al{al_percentage}')
                        # the first value of budget_train is 100...
                        if total_budget == 100:
                            results[dataset][percentageString] = {}
                            results[dataset][percentageString]['recall'] = []
                            results[dataset][percentageString]['precision'] = []
                            results[dataset][percentageString]['f1'] = []
                        # Calculate metrics
                        results[dataset][percentageString]['recall'].append(
                            metrics.recall_score(y_test, y_pred_learner))
                        results[dataset][percentageString]['precision'].append(
                            metrics.precision_score(y_test, y_pred_learner))
                        results[dataset][percentageString]['f1'].append(
                            metrics.f1_score(y_test, y_pred_learner))
                        print(f"recall : {metrics.recall_score(y_test, y_pred_learner)} ")

            if conf == "block&al_standard_tesis":
                for total_budget in range(100, 600, 100):  # total_budget 100 and 500
                #for total_budget in range(100, 101, 10):  # total_budget is 100
                    for block_percentage in range(10, 110, 10):  # if tot_budget = 100 start on 10 and end in 100
                        # initialization
                        learner = ActiveLearner(
                            estimator=RandomForestClassifier(n_estimators=100),
                        )
                        X_non_used, y_non_used, X_used, y_used = getX_non_used_y_non_used_X_used_y_used(
                            dataset,
                            int(block_percentage / 100 * total_budget),
                            classifiers[0]['name'],
                            query_strategies[0]['name'],
                            blk.copy(),
                            sim_config
                        )
                        al_percentage = 100 - block_percentage
                        # if one of the percentages is equal to 0 the function is a little bit different
                        if block_percentage == 100:
                            # use 100% from the blocker data
                            X_test = X_non_used.copy()
                            y_test = y_non_used.copy()
                            learner.teach(X_used, np.ravel(y_used))
                            # al choose from all the samples

                        else:

                            print(f"blk% {block_percentage}, al% {al_percentage}")
                            print(f"X used {X_used.shape}")
                            print(f"X non_used {X_non_used.shape}")

                            # Training the learner using the data already used by the Block Al
                            learner.teach(X_used, np.ravel(y_used))

                            X_al_pool = X_non_used.copy()
                            y_al_pool = y_non_used.copy()

                            # Training the learner letting he choose from all the pairs remaining
                            for n in range(int((total_budget * (al_percentage / 100)) / q_num)):
                                query_idx, query_inst = learner.query(X_al_pool, n_instances=q_num)
                                y_new = y_al_pool.loc[query_inst.index]
                                try:
                                    learner.teach(query_inst, np.ravel(y_new))
                                except:
                                    print("Error in al percentage for in index: ")
                                X_al_pool = X_al_pool.drop(index=query_inst.index)
                                y_al_pool = y_al_pool.drop(index=query_inst.index)

                            X_test = X_al_pool.copy()
                            y_test = y_al_pool.copy()

                        y_pred_learner = learner.predict(X_test)
                        percentageString = str(f'block{block_percentage}_al{al_percentage}')
                        # the first value of budget_train is 100...
                        if total_budget == 100:
                            results[dataset][percentageString] = {}
                            results[dataset][percentageString]['recall'] = []
                            results[dataset][percentageString]['precision'] = []
                            results[dataset][percentageString]['f1'] = []
                        # Calculate metrics
                        results[dataset][percentageString]['recall'].append(
                            metrics.recall_score(y_test, y_pred_learner))
                        results[dataset][percentageString]['precision'].append(
                            metrics.precision_score(y_test, y_pred_learner))
                        results[dataset][percentageString]['f1'].append(
                            metrics.f1_score(y_test, y_pred_learner))
                        print(f"recall : {metrics.recall_score(y_test, y_pred_learner)} ")
        with open(f"{conf}_metrics_dict.pickle", "wb") as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
