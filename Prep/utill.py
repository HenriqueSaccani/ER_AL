'''
author: giovanni.simonini@gmail.com
'''
import pandas as pd
import string
from typing import List, Iterator


class Metrics(object):
    """
    Just compute:
    Recall
    Precision
    F-score
    """

    gt_set = None
    __recall = None
    __precison = None

    def __init__(self, df_gt):
        self.gt_set = set()
        self.__recall = None
        self.__precison = None
        for i, row in df_gt.iterrows():
            self.gt_set.add((row.loc['id_l'], row.loc['id_r']))

    def getGTSize(self):
        return len(self.gt_set)

    def getRecall(self, pairs_list: List) -> float:
        if not self.__recall:
            print('compute recall')
            pairs_set = set(pairs_list)
            true_positive = self.gt_set.intersection(pairs_set)
            self.__recall = len(true_positive) / len(self.gt_set)
        return self.__recall

    def getPrecision(self, pairs_list: List) -> float:
        if not self.__precison:
            pairs_set = set(pairs_list)
            true_positive = self.gt_set.intersection(pairs_set)
            self.__precison = len(true_positive) / len(pairs_list)
        return self.__precison

    def getFScore(self, pairs_list: List) -> float:
        r = self.getRecall(pairs_list)
        p = self.getPrecision(pairs_list)
        return (2 * r * p) / (r + p)

    def resetMetrics(self):
        self.__recall = None
        self.__precison = None
        # print("gt: "+str(len(self.gt_set)))


class Block(object):
    def __init__(self):
        block = List[dict]


class TokenBlockingSchema(object):
    '''
    Takes as input two pandas dataframes:
    - schemas must be aligned (i.e, same names)

    Creates a block for each token in each value:
    - a candidate pair is generated is two records appear together in a block
    '''
    blocks = dict()
    df1: pd.DataFrame
    df2: pd.DatetimeIndex

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame):
        self.df1 = df1
        self.df2 = df2
        self.blocks = dict()

    def __index__(self, df: pd.DataFrame, blocks, first):
        dataset = 0 if first else 1
        attributes = df.columns
        table = str.maketrans(dict.fromkeys(string.punctuation))
        for i, row in df.iterrows():
            keys = []
            for c, col in enumerate(attributes):
                value = row[col]
                if value:
                    bk = list(map(lambda x: x + "_" + str(c), str(value).translate(table).lower().split()))
                    keys.extend(bk)
            for k in keys:
                hk = hash(k)
                entry = blocks.get(hk, [[], []])
                entry[dataset].append(i)
                blocks[hk] = entry

    def index(self):
        self.__index__(self.df1, self.blocks, True)
        self.__index__(self.df2, self.blocks, False)

    def get_pair_iterator(self, max_comparisons) -> Iterator:
        '''
        :param max_comparisons: maximum number of comparison per block (i.e., if a block yields more than this number of comparisons is discarded)
        :return: Candidate pari iterator
        '''
        keys = list(self.blocks.keys())
        for k in keys:
            b0 = self.blocks.get(k)[0]
            b1 = self.blocks.get(k)[1]
            if len(b0) * len(b1) == 0 or len(b0) * len(b1) > max_comparisons:
                continue
            for r0 in b0:
                for r1 in b1:
                    yield (r0, r1)
