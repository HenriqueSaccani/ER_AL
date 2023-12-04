import os
import sys
import pandas as pd
from Prep import features_utills
import importlib
import time
from Couples import *
import numpy as np
import json
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

'''
Used to Create the dataframes with the similarities for each database and budget. budget used are 100 and 500.
This code create tbe dataframes using the already created couples (See "CouplesCreator" code), the couples calculated 
are 10,20,30...100 when budget is 100 and 50,100,150...500 when budget is 500. 
'''
datasets = {
            #"abtBuy": [[2, 3, 7, 8, 9, 10,12,15], [9,15], [14]],
            #"DblpAcm": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
            #"scholarDblp": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
            "amazonGoogleProducts": [[7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [14]]
}

if __name__ == '__main__':
    for dataset, sim_config in datasets.items():
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
            # id_l = idAmazon, id_r = url google
            gt.columns = ['id_l', 'id_r']

        blk = pd.read_parquet("../" + "Couples/" + dataset)

        df_total_features = pd.DataFrame(index=[], columns=[])

        for budget_train in range(50, 550, 50):
        #for budget_train in range(10, 101, 10):
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

            # Read existing file of calculated similarities or Calculate similarities
            sim_config_string = ""
            for v in sim_config:
                sim_config_string += f"{str(v)}_"

            file_df_features_name = f'df_features/df_features_{dataset}_{sim_config_string}{budget_train}.csv'
            file_df_used_features_name = f'df_features/df_used_features_{dataset}_{sim_config_string}{budget_train}.csv'

            #if file exists don't need to create another file
            if os.path.isfile(file_df_features_name) and os.path.isfile(file_df_used_features_name):
                df_features = features_utills.getDf_featuresFromFile(file_df_features_name, 'id_l', 'id_r')
                df_used_features = features_utills.getDf_featuresFromFile(file_df_used_features_name, 'id_l', 'id_r')
                df_total_features = pd.concat([df_features, df_used_features], axis=0)
                print(f"{file_df_features_name} exist ")
            else:
                # Use the already calculated pairs to speed up the calculation of others pairs
                # df_total_features is a df of all the already calculated pairs
                if not df_total_features.index.empty:
                    # find the missing pairs to calculate
                    pairs_not_calculated = []
                    for pair in pairs:
                        if pair not in df_total_features.index:
                            pairs_not_calculated.append(pair)

                    for used_pair in used_pairs:
                        if used_pair not in df_total_features.index:
                            pairs_not_calculated.append(used_pair)
                    #concatenate the non calculated pairs and call the function to calculate them
                    col_names = features_utills.getFeaturesNames(df1.columns, sim_config, True)
                    uncompleted_pairs_sim = features_utills.parallelGetColSim(df1, df2, pairs_not_calculated,
                                                                              sim_config, True, 8)
                    #Create a dataframe to concat with the existing one and then extrat the pairs needed
                    df_unc_features = pd.DataFrame(uncompleted_pairs_sim, columns=col_names)
                    df_unc_features.set_index(['id_l', 'id_r'], inplace=True)
                    df_unc_features['label'] = 0
                    #Insert correct label values
                    for p in list(zip(gt.id_l.values, gt.id_r.values)):
                        if p in df_unc_features.index:
                            df_unc_features.loc[p, 'label'] = 1

                    df_total_features = pd.concat([df_total_features, df_unc_features], axis=0)
                    try:
                        df_features = df_total_features.loc[[pair for pair in pairs]]
                        df_used_features = df_total_features.loc[[used_pair for used_pair in used_pairs]]
                    except:
                        print(df_features.isna().any())
                #if df_total_features is empty computate normaly all the pairs features
                else:
                    col_names = features_utills.getFeaturesNames(df1.columns, sim_config, True)
                    # similarities of non labeled data
                    pairs_sim = features_utills.parallelGetColSim(df1, df2, pairs, sim_config, True, 8)
                    used_pairs_sim = features_utills.parallelGetColSim(df1, df2, used_pairs, sim_config,
                                                                       True, 8)
                    # Compute all the similarities in the function
                    df_features = pd.DataFrame(pairs_sim, columns=col_names)
                    df_features.set_index(['id_l', 'id_r'], inplace=True)

                    df_used_features = pd.DataFrame(used_pairs_sim, columns=col_names)
                    df_used_features.set_index(['id_l', 'id_r'], inplace=True)

                    df_features['label'] = 0
                    df_used_features['label'] = 0
                    for p in list(zip(gt.id_l.values, gt.id_r.values)):
                        if p in df_features.index:
                            df_features.loc[p, 'label'] = 1
                        if p in df_used_features.index:
                            df_used_features.loc[p, 'label'] = 1
                    df_total_features = pd.concat([df_features, df_used_features], axis=0)

                # Save Df_features in a csv file

                df_features.to_csv(f"df_features/df_features_{dataset}_{sim_config_string}{budget_train}.csv")
                df_used_features.to_csv(f"df_features/df_used_features_{dataset}_{sim_config_string}{budget_train}.csv")
