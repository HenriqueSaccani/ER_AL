from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import concurrent.futures
import time

'''
This Dataset create possible Entity matching couples depending on the number of budget used to train a Blocking schema.
In this project we are using this budgets to train the Active learning Blocking schema for any dataset:
10-20-30-40-50-60-70-80-90-100-150-200-250-300-350-400-450-500.
This couples are using later in the "df_features_creator" to create similarity dataframes who are used later 
on another code , see "MetricsCalculator" to finally train another learner and cal
Using a bigger budget normally means a bigger recall and a smaller precision.
'''


# datasets = ["abtBuy", "DblpAcm", "scholarDblp", "amazonGoogleProducts"]
datasets = {"amazonGoogleProducts"}

query_strategies = [
    # {"strategy": entropy_sampling, "name": "entropy_sampling", "color": "#234f1d"},
    # {"strategy": margin_sampling, "name": "margin_sampling", "color": "#0000ff"},
    {"strategy": uncertainty_sampling, "name": "uncertainty_sampling", "color": "#ff00d8"}
]

classifiers = [
    # {"model": GaussianNB, "name": "naive_bayes"},
    # {"model": SVC, "name": "svm"},
    {"model": RandomForestClassifier, "name": "random_forest"}
]

configurations = [
    {"confName": "C12",
     "features": ["CFIBF", "RACCB", "JS", "nonRedudantComparisonsP1", "nonRedudantComparisonsP2", "isMatch"]}
]

# Number of records to label in each AL iteration
query_num = 1
# Number of samples for the training set
number_of_samples = 200
# Number of samples to train the model for AL, randomly extracted from the training set
n_initial = 10


# Given a dataframe extracts n samples equally divided in positives and negatives
def get_train_test(df, n_samples, label, seed, pos_value=1, neg_value=0):
    # Extracts the matching records
    match = df[df[label] == pos_value]
    # Extracts the non-matching records
    non_match = df[df[label] == neg_value]

    # Takes n/2 samples from the matching records
    pos_train = match.sample(n=int(n_samples / 2), replace=False, random_state=seed)
    # Removes the extracted samples, these are used for testing
    pos_test = match.drop(index=pos_train.index)
    # To have a balanced training set, takes the same number of samples of the matching records
    neg_train = non_match.sample(n=len(pos_train), replace=False, random_state=seed)
    # Removes the extracted samples, these are used for testing
    neg_test = non_match.drop(index=neg_train.index)

    # Training set
    train_norm = pd.concat([pos_train, neg_train], axis=0)
    X_train = train_norm.drop(label, axis=1)
    y_train = train_norm[[label]]

    # Test set
    test = pd.concat([pos_test, neg_test], axis=0)
    X_test = test.drop(label, axis=1)
    y_test = test[[label]]

    return X_train, X_test, y_train, y_test


# Computes precision and recall
def calc_prec_rec(predictions, labels):
    cm = confusion_matrix(predictions, labels)
    tn = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tp = cm[1][1]

    rec = 0
    prec = 0
    f1 = 0
    if (tp + fn) != 0:
        rec = tp / (tp + fn)

    if (tp + fp) != 0:
        prec = tp / (tp + fp)

    if (prec + rec) != 0:
        f1 = (2 * prec * rec) / (prec + rec)

    return [rec, prec, f1]


# Save the results in a text file
def saveResults(outName, results):
    data = json.dumps(results)
    out = open(outName + ".txt", "wt")
    out.write(data)
    out.close()
    pass


def saveUsedData(outName, results):
    results.to_csv(outName + ".csv")


# Given a budget tests the classifier performances
def testClassifier(classifier, X, budget_train, query_num, query_strategies, n_initial, seed, dataset):
    test_res = {}

    # First test the classifier on a random sample
    X_train, X_test, y_train, y_test = get_train_test(X.copy(), budget_train, "isMatch", seed)

    est = None
    if classifier["name"] == "svm":
        est = classifier["model"](probability=True, kernel='linear')
    else:
        est = classifier["model"]()

    model = est.fit(X_train, np.ravel(y_train))
    test_res["random"] = calc_prec_rec(model.predict(X_test), y_test)

    # Extracts the initial samples for AL
    X_initial, X_pool, y_initial, y_pool = get_train_test(X.copy(), n_initial, "isMatch", seed)
    # Number of iterations
    iter_num = int((budget_train - n_initial) / query_num)

    # For each query strategy
    for q in query_strategies:

        X_pool_int = X_pool.copy()
        y_pool_int = y_pool.copy()

        est = None
        if classifier["name"] == "svm":
            est = classifier["model"](probability=True, kernel='linear')
        else:
            est = classifier["model"]()

        # Initializes the learner
        learner = ActiveLearner(
            estimator=est,
            X_training=X_initial,
            y_training=np.ravel(y_initial),
            query_strategy=q["strategy"]
        )
        used_indexes = []
        for i in range(iter_num):
            q_num = query_num
            # Asks records to label
            query_idx, query_inst = learner.query(X_pool_int, n_instances=q_num)
            # print(query_inst.columns)
            # Gets the labels for the requested records
            y_new = y_pool_int.loc[query_inst.index]
            # Teaches the model, i.e. provides the value of the requested record
            learner.teach(query_inst, np.ravel(y_new))
            # Removes the provided record from the pool
            X_pool_int = X_pool_int.drop(index=query_inst.index)
            y_pool_int = y_pool_int.drop(index=query_inst.index)
            used_indexes += list(query_inst.index)

        # Preparing X_used to be saved
        # X_used = X_pool.merge(X_pool_int, how="outer", indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        # X_used.drop(columns='_merge', axis=1, inplace=True)
        X_used = X_pool.loc[np.ravel(used_indexes)]
        X_used = pd.concat([X_used, X_initial])
        # Including label
        X_used['label'] = 0
        for x in X_used.index.values:
            if x in y_initial.index.values:
                if y_initial.loc[x]['isMatch'] == 1:
                    X_used.loc[x, "label"] = 1
            if x in y_pool.index.values:
                if y_pool.loc[x]['isMatch'] == 1:
                    X_used.loc[x, "label"] = 1
        print(f"saving X_used {len(X_used)} and ini:{len(X_initial)} pool:{len(X_pool)}, poll_int {len(X_pool_int)}...")
        saveUsedData(f"already_labeled_{dataset}_{budget_train}_{classifier['name']}_{q['name']}",
                     X_used)  # Save used instances
        # Saving couples
        X_pool_int["prediction"] = learner.predict(X_pool_int)
        possible_couples = X_pool_int.loc[X_pool_int["prediction"] == 1].drop(columns="prediction", axis=1)
        X_pool_int.drop(columns="prediction", axis=1, inplace=True)
        saveUsedData(f"possible_couples_{dataset}_{budget_train}_{classifier['name']}_{q['name']}", possible_couples)

        # Evaluates the model after providing new records
        pred = learner.predict(X_pool_int)
        test_res[q["name"]] = calc_prec_rec(pred, y_pool_int)

    return test_res


def launchProc(dataset, df, conf, run):
    global classifiers
    X = df[conf["features"]]
    results = {}
    for c in classifiers:
        results[c["name"]] = []
        if run == 500:
            for budget in range(50, 550, 50):  # if budget is 500
                d = {"budget": budget}
                d["results"] = testClassifier(c, X, budget, query_num, query_strategies, n_initial, run, dataset)
                results[c["name"]].append(d)
        elif run == 100:
            for budget in range(10, 101, 10):  # if budget is 500
                d = {"budget": budget}
                d["results"] = testClassifier(c, X, budget, query_num, query_strategies, n_initial, run, dataset)
                results[c["name"]].append(d)

    saveResults(dataset + "_" + conf["confName"] + "_run" + str(run), results)




for dataset in datasets:
    # Loads the data
    df = pd.read_parquet("../" + "Couples/" + dataset)
    t = 0
    for n, conf in enumerate(configurations):
        t1 = time.time()
        launchProc(dataset, df, conf, 100)
        t1 = time.time() - t1
        t += t1
        print(f"{conf} took :{t1} s")
    print(f"Couples for dataset {dataset}  were saved in {t}")
