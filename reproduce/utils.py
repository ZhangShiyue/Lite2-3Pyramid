import sys
sys.path.insert(0, "..")
import os
import numpy as np
import pickle as pkl
import xgboost as xgb
from itertools import groupby
from scipy.stats import pearsonr, spearmanr
from nltk import word_tokenize
from allennlp.predictors.predictor import Predictor
from metric import _run_srl, _get_srl_list, _get_features


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


def mix_scus_stus_by_folds(
        scus,
        percentage,
        references=None,
        ref_srls_pkl=None,
        ref_corefs=None,
        output_dir=None,
        use_coref=False,
        device=-1
):
    assert references is not None or ref_srls_pkl is not None, \
        "need to provide either references or the address of ref_srls.pkl"

    print(f"===Get doc ids by folds===")
    with open(f"../data/REALSumm/ids.txt", 'r') as f:
        doc_ids = [line.strip() for line in f.readlines()]

    did_fold_map = {}
    for fold in range(5):
        with open(f"../data/REALSumm/by_examples/fold{fold + 1}.id", 'r') as f:
            for line in f.readlines():
                did_fold_map[line.strip()] = fold

    if ref_srls_pkl:
        with open(ref_srls_pkl, 'rb') as f:
            ref_srls = pkl.load(f)
        if use_coref:
            with open(ref_corefs, 'rb') as f:
                ref_corefs = pkl.load(f)
    else:
        res = _run_srl(references, doc_ids, output_dir=output_dir, use_coref=use_coref)
        ref_srls = res[0]
        if use_coref: ref_corefs = res[1]

    print(f"===Get Parsing Trees===")
    # get parsing trees
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz",
        cuda_device=device)

    all_trees, all_scus, all_stus = {}, {}, {}
    for doc_index, doc_id in enumerate(doc_ids):
        # get features
        trees = []
        for sent in ref_srls[doc_id]:
            sent = ' '.join(sent["words"])
            res = predictor.predict(sentence=sent)
            trees.append([res["tokens"], res["trees"]])
        all_trees[doc_id] = trees

        # align scus
        sent_words = [[word.lower() for word in sent["words"]] for sent in ref_srls[doc_id]]
        scu_sent = {}
        for scu in scus[doc_index]:
            max_sent, max_overlap = 0, 0
            scu_word = word_tokenize(scu.lower().replace(',', ' ').replace('.', ' '))
            for i, sent_word in enumerate(sent_words):
                overlap = len(set(scu_word) & set(sent_word))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_sent = i
            if max_sent not in scu_sent:
                scu_sent[max_sent] = []
            scu_sent[max_sent].append(scu)
        all_scus[doc_id] = scu_sent

        # align stus
        srl_sent = {}
        if use_coref:
            for srl in ref_corefs[doc_id][1]:
                srl_word = word_tokenize(srl.lower().split("is")[1])
                max_sent, max_overlap = 0, 0
                for i, sent_word in enumerate(sent_words):
                    overlap = len(set(srl_word) & set(sent_word))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_sent = i
                if max_sent not in srl_sent:
                    srl_sent[max_sent] = []
                srl_sent[max_sent].append(srl)
        for i, sent in enumerate(ref_srls[doc_id]):
            if i not in srl_sent:
                srl_sent[i] = []
            srl_sent[i].extend(_get_srl_list([sent]))
        all_stus[doc_id] = srl_sent

    print(f"===Get Easiness Scores by folds===")
    # predict easiness scores
    easinesses = []
    easiness_dict = {}
    for fold in range(5):
        X, dids, sids = [], [], []
        for doc_id in doc_ids:
            if did_fold_map[doc_id] != fold:
                continue
            for sent_id, (tokens, tree) in enumerate(all_trees[doc_id]):
                X.append(_get_features(tokens, tree))
                dids.append(doc_id)
                sids.append(sent_id)
        X = xgb.DMatrix(np.array(X))
        bst = xgb.Booster({'nthread': 4})
        bst.load_model(f"../regressors/REALSumm/fold{fold}_xgb.json")
        scores = bst.predict(X)

        for did, sid, score in zip(dids, sids, scores):
            if did not in easiness_dict:
                easiness_dict[did] = {}
            easiness_dict[did][sid] = score
            easinesses.append(score)

    print(f"===Mix STUs and SCUs===")
    # predict easiness scores
    # mix scus and stus
    easinesses = sorted(easinesses, reverse=True)
    threshold = easinesses[:int(len(easinesses) * percentage / 100)][-1]

    units = []
    for doc_id in doc_ids:
        doc_units = []
        for sent in easiness_dict[doc_id]:
            if sent not in all_scus[doc_id]:
                continue
            easiness = easiness_dict[doc_id][sent]
            if easiness > threshold:
                doc_units.extend(all_stus[doc_id][sent])
            else:
                doc_units.extend(all_scus[doc_id][sent])
        units.append(doc_units)

    if output_dir:
        print(
            f"===Save STUs_SCUs_percentage{percentage} to {output_dir}/STUs_SCUs_percentage{percentage}_by_folds.txt===")
        with open(f"{output_dir}/STUs_SCUs_percentage{percentage}_by_folds.txt", 'w') as f:
            f.write('\n'.join(['\t'.join(us) for us in units]))

    return units


def get_realsumm_data(version=2):
    if version == 2:
        with open("../data/REALSumm/SCUs.txt", 'r') as f:
            units = [line.strip().split('\t') for line in f.readlines()]
    elif version == 3:
        with open("../data/REALSumm/STUs.txt", 'r') as f:
            units = [line.strip().split('\t') for line in f.readlines()]
    elif 2 < version < 3:
        percentage = int((version - 2) * 100)
        if not os.path.exists(f"../data/REALSumm/STUs_SCUs_percentage{percentage}_by_folds.txt"):
            print(f"===Prepare STUs_SCUs_percentage{percentage}_by_folds.txt")
            with open("../data/REALSumm/SCUs.txt", 'r') as f:
                scus = [line.strip().split('\t') for line in f.readlines()]
            mix_scus_stus_by_folds(scus=scus, percentage=percentage,
                                   ref_srls_pkl="../data/REALSumm/ref_srls.pkl",
                                   ref_corefs="../data/REALSumm/ref_corefs.pkl",
                                   output_dir="../data/REALSumm", use_coref=True)
        with open(f"../data/REALSumm/STUs_SCUs_percentage{percentage}_by_folds.txt", 'r') as f:
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


if __name__ == '__main__':
    with open("../data/REALSumm/SCUs.txt", 'r') as f:
        scus = [line.strip().split('\t') for line in f.readlines()]
    with open("../data/REALSumm/ids.txt", 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    mix_scus_stus_by_folds(scus=scus, percentage=50,
                  ref_srls_pkl="../data/REALSumm/ref_srls.pkl",
                  ref_corefs="../data/REALSumm/ref_corefs.pkl",
                  output_dir="../data/REALSumm", use_coref=True, device=0)