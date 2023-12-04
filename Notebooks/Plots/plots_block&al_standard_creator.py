# %%

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import rcParams
import pickle
import pprint

''' This code create the standard plots, used in the abstract paper.
    The plots are divided by 3 types:
    dataset: A tsplot for each dataset, showing the standard variation 
            of 100 and 500 budget in the specific dataset. (4 in total)
    dataset_and_budget: A normal plot for each dataset and budgeet 
                        (4x2=8 in total) 
    group: A tsplot for each group, the 4 datasets are divided in 2 groups.
            (2 plots in total)
    group_and_budget : A tsplot for each group and budget.
                        (2*2=4 plots in total)
'''

datasets = {"abtBuy": [[2, 3, 7, 8, 9, 10, 12, 15], [9, 15], [14]],
            "DblpAcm": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
            "scholarDblp": [[3, 4, 8, 9, 10, 12], [14], [3, 4, 8, 9, 10, 12], [3, 4, 8, 9, 10, 12]],
            "amazonGoogleProducts": [[7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [7, 8, 9, 10, 12, 15], [14]]
            }
datasets_names_string = ''
for x in datasets.keys():
    datasets_names_string += x
    datasets_names_string += '_'
rec_color = 'blue'
pre_color = 'darkorange'
f1_color = 'green'
rcParams.update({'font.size': 17.5})
#x_label = block_al_dict['abtBuy'].keys()
x_label = ["10:90","20:80","30:70","40:60","50:50","60:40","70:30","80:20","90:10","100:0"]
#sns.set(style="white")
#sns.set()
def tsplot(data, **kw):
    x = np.arange(data.shape[0])
    est = np.mean(data, axis=1)
    sd = np.std(data, axis=1)
    cis = (est - sd, est + sd)
    kw_except_label = kw.copy()
    if 'label' in kw.keys():
        del kw_except_label['label']
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, **kw_except_label)
    plt.plot(est, **kw)

def defaultPlot():
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(8, 8))
    plt.ylim(0, 1)
    plt.xlabel("blocking:matching  budget ratio")
    plt.xticks(np.arange(10), x_label, rotation=25)

with open('../importanceAndFeatures_Pickles/block&al_standard_metrics_dict.pickle', 'rb') as handle:
    block_al_dict = pickle.load(handle)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(block_al_dict)

# Plots per database and budget
for dataset in datasets.keys():
    # one per dataset
    organized_rec_dict = {"100": [],
                          "500": []
                          }
    organized_pre_dict = {"100": [],
                          "500": []
                          }
    organized_f1_dict = {"100": [],
                         "500": []
                         }
    recall_data = []
    precision_data = []
    f1_data = []
    for k, v in block_al_dict[dataset].items():
        for i, budget_used in enumerate(organized_rec_dict.keys()):
            organized_rec_dict[budget_used].append(v['recall'][i])
            organized_pre_dict[budget_used].append(v['precision'][i])
            organized_f1_dict[budget_used].append(v['f1'][i])
        recall_data.append(v['recall'])
        precision_data.append(v['precision'])
        f1_data.append(v['f1'])

    pp.pprint(f1_data) ## pretty print F1

    for n, bud in enumerate(organized_rec_dict.keys()):
        defaultPlot()
        plt.plot(organized_rec_dict[str(bud)], label='recall', color=rec_color)
        plt.plot(organized_pre_dict[str(bud)], label='precision', color=pre_color)
        plt.plot(organized_f1_dict[str(bud)], label='f1', color=f1_color)

        plt.title(f"{dataset}-budget={bud}")

        plt.legend()
        plt.grid(True)

        plt.savefig(f"standard/{dataset}_{bud}.png")
        plt.show()

#plots per dataset only
for dataset in datasets.keys():
    # one per dataset
    organized_rec_dict = {"100": [],
                          "500": []
                          }
    organized_pre_dict = {"100": [],
                          "500": []
                          }
    organized_f1_dict = {"100": [],
                         "500": []
                         }
    recall_data = []
    precision_data = []
    f1_data = []
    for k, v in block_al_dict[dataset].items():
        for i, budget_used in enumerate(organized_rec_dict.keys()):
            organized_rec_dict[budget_used].append(v['recall'][i])
            organized_pre_dict[budget_used].append(v['precision'][i])
            organized_f1_dict[budget_used].append(v['f1'][i])
        recall_data.append(v['recall'])
        precision_data.append(v['precision'])
        f1_data.append(v['f1'])

    defaultPlot()
    tsplot(np.array(list(organized_rec_dict.values())).transpose(), label='recall', color=rec_color)
    tsplot(np.array(list(organized_pre_dict.values())).transpose(), label='precision', color=pre_color)
    tsplot(np.array(list(organized_f1_dict.values())).transpose(), label='f1', color=f1_color)

    plt.title(f"{dataset}")

    plt.legend()
    plt.grid(True)

    plt.savefig(f"standard/{dataset}.png")
    plt.show()

# Plots per group, group1=(abtBuy + amazonGoogle) and group2=(ACM and Scholar)
#group1:
grouped_rec_dict = {}
grouped_pre_dict = {}
grouped_f1_dict = {}
for k, v in block_al_dict[list(datasets.keys())[0]].items():
    grouped_rec_dict[k] = []
    grouped_pre_dict[k] = []
    grouped_f1_dict[k] = []


for dataset in ["abtBuy", "amazonGoogleProducts"]:
    for k, v in block_al_dict[dataset].items():

        grouped_rec_dict[k] = grouped_rec_dict.get(k) + v['recall']
        grouped_pre_dict[k] = grouped_pre_dict.get(k) + v['precision']
        grouped_f1_dict[k] = grouped_f1_dict.get(k) + v['f1']

defaultPlot()
tsplot(np.array(list(grouped_rec_dict.values())), label='recall', color=rec_color)
tsplot(np.array(list(grouped_pre_dict.values())), label='precision', color=pre_color)
tsplot(np.array(list(grouped_f1_dict.values())), label='f1', color=f1_color)

plt.title(f"abtBuy+amazonGoogleProducts")
plt.legend()
plt.grid(True)
plt.savefig(f"standard/abtBuy+amazonGoogleProducts.png")
plt.show()

#group2
grouped_rec_dict = {}
grouped_pre_dict = {}
grouped_f1_dict = {}
for k, v in block_al_dict[list(datasets.keys())[0]].items():
    grouped_rec_dict[k] = []
    grouped_pre_dict[k] = []
    grouped_f1_dict[k] = []
for dataset in ["DblpAcm", "scholarDblp"]:
    for k, v in block_al_dict[dataset].items():
        grouped_rec_dict[k] = grouped_rec_dict.get(k) + v['recall']
        grouped_pre_dict[k] = grouped_pre_dict.get(k) + v['precision']
        grouped_f1_dict[k] = grouped_f1_dict.get(k) + v['f1']

defaultPlot()
tsplot(np.array(list(grouped_rec_dict.values())), label='recall', color=rec_color)
tsplot(np.array(list(grouped_pre_dict.values())), label='precision', color=pre_color)
tsplot(np.array(list(grouped_f1_dict.values())), label='f1', color=f1_color)

plt.title(f"DblpAcm+scholarDblp")
plt.legend()
plt.grid(True)
plt.savefig(f"standard/DblpAcm+scholarDblp.png")
plt.show()

# Plots per group and budget, group1=(abtBuy + amazonGoogle) and group2=(ACM and Scholar)

#group1:
for i in range(2):
    budget=100 if i == 0  else 500
    grouped_rec_dict = {}
    grouped_pre_dict = {}
    grouped_f1_dict = {}
    for k, v in block_al_dict[list(datasets.keys())[0]].items():
        grouped_rec_dict[k] = []
        grouped_pre_dict[k] = []
        grouped_f1_dict[k] = []

    for dataset in ["abtBuy", "amazonGoogleProducts"]:
        for k, v in block_al_dict[dataset].items():
            grouped_rec_dict[k].append(v['recall'][i])
            grouped_pre_dict[k].append(v['precision'][i])
            grouped_f1_dict[k].append(v['f1'][i])

    defaultPlot()
    tsplot(np.array(list(grouped_rec_dict.values())), label='recall', color=rec_color)
    tsplot(np.array(list(grouped_pre_dict.values())), label='precision', color=pre_color)
    tsplot(np.array(list(grouped_f1_dict.values())), label='f1', color=f1_color)

    plt.title(f"abtBuy+amazonGoogleProducts budget={budget}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"standard/abtBuy+amazonGoogleProducts_{budget}.png")
    plt.show()

#group2:
for i in range(2):
    budget=100 if i == 0  else 500
    grouped_rec_dict = {}
    grouped_pre_dict = {}
    grouped_f1_dict = {}
    for k, v in block_al_dict[list(datasets.keys())[0]].items():
        grouped_rec_dict[k] = []
        grouped_pre_dict[k] = []
        grouped_f1_dict[k] = []

    for dataset in ["DblpAcm", "scholarDblp"]:
        for k, v in block_al_dict[dataset].items():
            grouped_rec_dict[k].append(v['recall'][i])
            grouped_pre_dict[k].append(v['precision'][i])
            grouped_f1_dict[k].append(v['f1'][i])

    defaultPlot()
    tsplot(np.array(list(grouped_rec_dict.values())), label='recall', color=rec_color)
    tsplot(np.array(list(grouped_pre_dict.values())), label='precision', color=pre_color)
    tsplot(np.array(list(grouped_f1_dict.values())), label='f1', color=f1_color)

    plt.title(f"DblpAcm+scholarDblp budget={budget}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"standard/DblpAcm+scholarDblp_{budget}.png")
    plt.show()
