import os
import numpy as np
from itertools import groupby
from scipy.stats import pearsonr, spearmanr


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def system_level_correlation(human, other, metric=None):
    system_human, system_other = [], []
    for system_name in human:
        human_score = np.mean([human[system_name][doc_id] for doc_id in human[system_name]])
        system_human.append(human_score)
        other_score = np.mean([other[system_name][doc_id][metric] if metric else other[system_name][doc_id]
                               for doc_id in human[system_name]])
        system_other.append(other_score)
    return pearsonr(system_human, system_other)[0], spearmanr(system_human, system_other)[0]


def summary_level_correlation(human, other, metric=None):
    summary_human, summary_other = {}, {}
    for system_name in human:
        for doc_id in human[system_name]:
            if doc_id not in summary_human:
                summary_human[doc_id] = []
                summary_other[doc_id] = []
            summary_human[doc_id].append(human[system_name][doc_id])
            summary_other[doc_id].append(other[system_name][doc_id][metric] if metric else other[system_name][doc_id])
    corrs_pear, corrs_spear, corrs_kend = [], [], []
    for doc_id in summary_human:
        for system_x in [summary_human, summary_other]:
            if all_equal(system_x[doc_id]):
                system_x[doc_id][0] += 1e-10
        corrs_pear.append(pearsonr(summary_human[doc_id], summary_other[doc_id])[0])
        corrs_spear.append(spearmanr(summary_human[doc_id], summary_other[doc_id])[0])
    return np.mean(corrs_pear), np.mean(corrs_spear)


def get_realsumm_data(version=2):
    if version == 2:
        with open("../data/REALSumm/SCUs.txt", 'r') as f:
            units = [line.strip().split('\t') for line in f.readlines()]
    elif version == 3:
        with open("../data/REALSumm/STUs.txt", 'r') as f:
            units = [line.strip().split('\t') for line in f.readlines()]

    system_data = {}
    for file in os.listdir("../data/REALSumm"):
        if "summary" in file:
            system_name = file.split('.')[0]

            with open(f"../data/REALSumm/{system_name}.summary", 'r') as f:
                summaries = [line.strip() for line in f.readlines()]
            with open(f"../data/REALSumm/{system_name}.label", 'r') as f:
                labels = [[int(label) for label in line.strip().split('\t')] for line in f.readlines()]

            system_data[system_name] = [summaries, labels]

    return units, system_data