from tqdm import tqdm
import textdistance as tt
import pandas as pd
import concurrent.futures
import math


def AdjustYear(x):
    if 100 > x > 20:
        x = 1900 + x
    elif x < 20:
        2000 + x
    return x


def sim_chooserController(sim_chooser, dif_sim_per_column=False):
    if sim_chooser is None and not dif_sim_per_column:
        sim_chooser = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    if sim_chooser is not None and dif_sim_per_column:
        try:
            for n, col in enumerate(sim_chooser):
                if type(col) is not list:
                    sim_chooser[n] = list(col)
                sim_chooser[n].sort()
        except:
            sim_chooser = [[]]
            print("invalid sim_chooser")

    if type(sim_chooser) is not list:
        try:
            sim_chooser = [sim_chooser]
        except:
            print("wrong sim_chooser declaration, should be in list type")
            sim_chooser = []

    if not dif_sim_per_column:
        sim_chooser.sort()

    return sim_chooser


def getFeaturesNames(columns, sim_chooser=None, dif_sim_per_column=False):
    '''
    Have in input the collum names of the database and return an array
    with all the names of the features who will be used in the matching.
    The textdistance library have a function to each of this similarities
    1-Hamming 2-Levenshtein 3-Damerau levenshtein 4-Needleman-Wunsch 5-Gotoh 6-Smith-Waterman
    7-Jaccard 8-Tversky index 9-Overlap coefficient 10-Cosine 11-Monge-Elkan 12-Bag distance
    13-Arithmetic coding 14 numerical distance  15-Ratcliff-Obershelp similarity
    '''
    sim_chooser = sim_chooserController(sim_chooser, dif_sim_per_column)
    col_names = ['id_l', 'id_r']
    if not dif_sim_per_column:
        for a in columns:
            # Edit based
            if 1 in sim_chooser:
                col_names.append(a + featureMapper(1))  # Hamming
            # col_names.append(a+'_mli') # Mlipns
            if 2 in sim_chooser:
                col_names.append(a + featureMapper(2))  # Levenshtein
            # col_names.append(a+'_S95') # Strcmp95
            # col_names.append(a+'_jW') # Jaro-Winkler
            if 3 in sim_chooser:
                col_names.append(a + featureMapper(3))  # damerau levenshtein
            if 4 in sim_chooser:
                col_names.append(a + featureMapper(4))  # Needleman-Wunsch
            if 5 in sim_chooser:
                col_names.append(a + featureMapper(5))  # Gotoh
            if 6 in sim_chooser:
                col_names.append(a + featureMapper(6))  # Smith-Waterman
            # Token based
            if 7 in sim_chooser:
                col_names.append(a + featureMapper(7))  # jaccard
            # col_names.append(a+'_sD') # Sørensen–Dice coefficient
            if 8 in sim_chooser:
                col_names.append(a + featureMapper(8))  # Tversky index
            if 9 in sim_chooser:
                col_names.append(a + featureMapper(9))  # Overlap coefficient
            # col_names.append(a+'_td') #Tanimoto distance
            if 10 in sim_chooser:
                col_names.append(a + featureMapper(10))  # cosine
            if 11 in sim_chooser:
                col_names.append(a + featureMapper(11))  # Monge-Elkan
            if 12 in sim_chooser:
                col_names.append(a + featureMapper(12))  # Bag distance
            # Compression based
            if 13 in sim_chooser:
                col_names.append(a + featureMapper(13))  # Arithmetic coding
            # col_names.append(a+'_RLE') #RLE
            # col_names.append(a+'_BRLE') #BWT RLE
            # col_names.append(a+'_sqrt') #Square Root
            # if sim_chooser == 14 or sim_chooser == 0:
            # col_names.append(a + '_ent')  # Entropy
            # Phonetic
            # col_names.append(a+'_MRA') #MRA
            # col_names.append(a+'_edi') #Editex
            # Numerical only
            if 14 in sim_chooser:
                col_names.append(a + featureMapper(14))  # MaxMin division
            if 15 in sim_chooser:
                col_names.append(a + featureMapper(15))  # Ratcliff-Obershelp similarity
    else:
        for n, a in enumerate(columns):
            # Edit basedh
            if 1 in sim_chooser[n]:
                col_names.append(a + featureMapper(1))  # Hamming
            if 2 in sim_chooser[n]:
                col_names.append(a + featureMapper(2))  # Levenshtein
            if 3 in sim_chooser[n]:
                col_names.append(a + featureMapper(3))  # damerau levenshtein
            if 4 in sim_chooser[n]:
                col_names.append(a + featureMapper(4))  # Needleman-Wunsch
            if 5 in sim_chooser[n]:
                col_names.append(a + featureMapper(5))  # Gotoh
            if 6 in sim_chooser[n]:
                col_names.append(a + featureMapper(6))  # Smith-Waterman
            # Token based
            if 7 in sim_chooser[n]:
                col_names.append(a + featureMapper(7))  # jaccard
            if 8 in sim_chooser[n]:
                col_names.append(a + featureMapper(8))  # Tversky index
            if 9 in sim_chooser[n]:
                col_names.append(a + featureMapper(9))  # Overlap coefficient
            if 10 in sim_chooser[n]:
                col_names.append(a + featureMapper(10))  # cosine
            if 11 in sim_chooser[n]:
                col_names.append(a + featureMapper(11))  # Monge-Elkan
            if 12 in sim_chooser[n]:
                col_names.append(a + featureMapper(12))  # Bag distance
            # Compression based
            if 13 in sim_chooser[n]:
                col_names.append(a + featureMapper(13))  # Arithmetic coding
            # Numerical division
            if 14 in sim_chooser[n]:
                col_names.append(a + featureMapper(14))  # MaxMin division
            if 15 in sim_chooser[n]:
                col_names.append(a + featureMapper(15))  # Ratcliff-Obershelp similarity

    return col_names


def getColSim(df1: pd.DataFrame, df2: pd.DataFrame, pairs, sim_chooser=None, dif_sim_per_column=False):
    sim_chooser = sim_chooserController(sim_chooser, dif_sim_per_column)
    all_pairs_sim = []
    for p in tqdm(pairs):
        r0 = df1.loc[p[0]]
        r1 = df2.loc[p[1]]
        similarities = [p[0], p[1]]
        for n, a in enumerate(df1.columns):
            s0 = str(r0[a]).lower()
            s1 = str(r1[a]).lower()
            if dif_sim_per_column:
                sim = simCalculator(sim_chooser[n], s0, s1)
            else:
                sim = simCalculator(sim_chooser, s0, s1)
            for s in sim:
                similarities.append(s)

        all_pairs_sim.append(similarities)

    return all_pairs_sim


def simCalculator(sim_chooser, s0, s1):
    similarities = []
    if 1 in sim_chooser:
        # Hamming
        sim = tt.hamming.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 2 in sim_chooser:
        # Levenshtein
        sim = tt.levenshtein.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 3 in sim_chooser:
        # Damerau levenshtein # this is slow
        sim = tt.damerau_levenshtein.normalized_similarity(s0, s1)
        ##sim = pylev.damerau_levenshtein(s0,s1)
        similarities.append(sim)
    if 4 in sim_chooser:
        # Needleman-Wunsch
        sim = tt.needleman_wunsch.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 5 in sim_chooser:
        # Gotoh
        sim = tt.gotoh.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 6 in sim_chooser:
        # Smith-Waterman
        sim = tt.smith_waterman.normalized_similarity(s0, s1)
        similarities.append(sim)
    ###Token Based
    if 7 in sim_chooser:
        # Jaccard
        sim = tt.jaccard.normalized_similarity(s0, s1)
        similarities.append(sim)
    ##Sørensen–Dice coefficient
    # sim = tt.Sorensen.normalized_similarity(s0,s1)
    # similarities.append(sim)
    if 8 in sim_chooser:
        # Tversky index
        sim = tt.tversky.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 9 in sim_chooser:
        # Overlap coefficient
        sim = tt.overlap.normalized_similarity(s0, s1)
        similarities.append(sim)
    ##Tanimoto distance
    # sim = tt.Tanimoto.normalized_similarity(s0,s1)
    # similarities.append(sim)
    if 10 in sim_chooser:
        # Cosine
        sim = tt.cosine.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 11 in sim_chooser:
        # Monge-Elkan
        sim = tt.monge_elkan.normalized_similarity(s0, s1)
        similarities.append(sim)
    if 12 in sim_chooser:
        # Bag distance
        sim = tt.bag.normalized_similarity(s0, s1)
        similarities.append(sim)

    ###Compression Based
    if 13 in sim_chooser:
        # Arithmetic coding
        sim = tt.arith_ncd.normalized_similarity(s0, s1)
        similarities.append(sim)

    ##RLE
    # sim = tt.RLENCD.normalized_similarity(s0,s1)
    # similarities.append(sim)
    ##BWT RLE
    # sim = tt.BWTRLENCD.normalized_similarity(s0,s1)
    # similarities.append(sim)
    ##Square Root
    # sim = tt.sqrt_ncd.normalized_similarity(s0,s1)
    # similarities.append(sim)
    # if sim_chooser == 14 or sim_chooser == 0:
    # Entropy
    # sim = tt.entropy_ncd.normalized_similarity(s0, s1)
    # similarities.append(sim)
    if 14 in sim_chooser:
        # minMax division (only for numbers)
        try:
            sim = min(float(s0), float(s1)) / max(float(s0), float(s1))
            if math.isnan(sim):
                sim = 0
        except:
            sim = 0
        similarities.append(sim)
    ##Sequence based

    if 15 in sim_chooser:
        sim = tt.ratcliff_obershelp(s0, s1)
        similarities.append(sim)

    return similarities


def parallelGetColSim(df1: pd.DataFrame, df2: pd.DataFrame, pairs, sim_chooser=None
                      , dif_sim_per_column=False, process=4):
    all_pairs_sim = []
    list_of_pairs = list(pairs)
    aux = round(len(list_of_pairs) / process)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = []
        for p in range(process):
            if p == process - 1:
                results.append(executor.submit(getColSim, df1, df2, list_of_pairs[p * aux:], sim_chooser
                                               , dif_sim_per_column))
            else:
                results.append(executor.submit(getColSim, df1, df2, list_of_pairs[p * aux: (p + 1) * aux],
                                               sim_chooser, dif_sim_per_column))
        for f in concurrent.futures.as_completed(results):
            all_pairs_sim += f.result()
    return all_pairs_sim


def featureMapper(n):
    mapper = {
        1: '_ham',
        2: '_ls',
        3: '_dLs',
        4: '_NW',
        5: '_go',
        6: '_sW',
        7: '_js',
        8: '_tv',
        9: '_oc',
        10: '_cs',
        11: '_mE',
        12: '_bag',
        13: '_ath',
        14: '_div',
        15: '_rO'
    }
    return mapper.get(n)


def getDf_featuresFromFile(file_df_features, id_left_name, id_right_name, set_type='str'):
    df_features = pd.read_csv(file_df_features)
    df_features[id_left_name] = df_features[id_left_name].astype(set_type)
    df_features[id_right_name] = df_features[id_right_name].astype(set_type)
    df_features.drop_duplicates(inplace=True, subset=[id_left_name, id_right_name])
    df_features.set_index([id_left_name, id_right_name], inplace=True)
    df_features = df_features.sort_index()

    return df_features
