import numpy as np
import pickle as pkl
import xgboost as xgb
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize
from allennlp.predictors.predictor import Predictor


def _get_and_replace_coref(references, doc_ids, device):
    # get coreference for every summary
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
        cuda_device=device
    )
    all_corefs = {}
    for i, doc_id in enumerate(tqdm(doc_ids)):
        summary = references[i].replace('<t>', ' ').replace('</t>', ' ').strip()
        coref = predictor.predict(document=summary)
        all_corefs[doc_id] = coref

    # replace the coreference with the actual entity name
    ref_corefs_replaced = {}
    for did in all_corefs:
        words = all_corefs[did]["document"]
        clusters = all_corefs[did]["clusters"]
        all_pronouns = {}
        scus = []
        for cluster in clusters:
            pronouns, others = [], []
            for s, e in cluster:
                span = ' '.join(words[s:e + 1]).strip()
                if span.lower() in ["it", "he", "she", "they", "this", "i", "you", "her", "him",
                                    "their", "them", "his", "its", "my", "your"]:
                    pronouns.append([s, e])
                else:
                    others.append([s, e, span])
            if len(others):
                replacement = others[0][-1]  # use the first one as replacement
                unique_others = set()
                for s, e, other in others[1:]:
                    other_words = set(other.lower().split(' '))
                    replacement_words = set(replacement.lower().split(' '))
                    if other.lower() != replacement.lower() and other.lower() not in unique_others \
                            and other.lower().find(replacement.lower()) == -1 \
                            and len(other_words & replacement_words) / len(other_words) < 0.5:
                        scus.append(f"{replacement} is {other}")
                        unique_others.add(other.lower())
                    all_pronouns[s] = (e, replacement)
                if len(pronouns):
                    for s, e in pronouns:
                        all_pronouns[s] = (e, replacement)
        if all_pronouns:
            new_words = []
            i = 0
            while i < len(words):
                if i in all_pronouns:
                    new_words.append(all_pronouns[i][1])
                    i = all_pronouns[i][0] + 1
                else:
                    new_words.append(words[i])
                    i += 1
            words = new_words
        ref_corefs_replaced[did] = [' '.join(words), scus]
    return ref_corefs_replaced


def _get_srl_list(sents):
    """
    SRL frames to STUs
    """
    all_srl_list = []
    for sent in sents:
        srl_list = []
        words = sent["words"]
        for item in sent["verbs"]:
            tags = item["tags"]
            tag_dict = []
            prev_words, v_words, prev, prev_word, prev_v_word = [], [], True, None, None
            for word, tag in zip(words, tags):
                if tag != "O":
                    position, tag = tag.split('-', 1)
                    if tag == "V":
                        prev = False
                        v_words.append(word)
                        prev_v_word = prev_word
                    elif not prev:
                        if len(tag_dict) == 0 or tag_dict[-1][0] != tag:
                            tag_dict.append([tag, []])
                        tag_dict[-1][1].append(word)
                    else:
                        prev_words.append(word)
                prev_word = word
            if len(prev_words) == 0 or len(v_words) == 0:
                continue
            else:
                if prev_v_word == "to": continue
                if prev_v_word in ["am", "are", "is", "was", "were", "been"]:
                    v_words = [prev_v_word] + v_words
                prev_words = ' '.join(prev_words)
                verb = ' '.join(v_words)
                if verb in ["'", "will", "could", "can", "might", "should", "would"]: continue
                if tag_dict and tag_dict[0][0] == "ARGM-NEG":
                    verb = f"{verb} {' '.join(tag_dict[0][1])}"
                    tag_dict = tag_dict[1:]
                for key, key_words in tag_dict:
                    arg = ' '.join(key_words)
                    srl = f"{prev_words} {verb} {arg}"
                    repeat = False
                    for osrl in srl_list:
                        if osrl == srl:
                            repeat = True
                            break
                    if not repeat:
                        srl_list.append(srl)

        if len(srl_list) == 0:
            srl_list.append(' '.join(words))
        all_srl_list.extend(srl_list)
    return all_srl_list


def _run_srl(
    references,
    doc_ids=None,
    output_dir=None,
    use_coref=False,
    device=-1,
):
    # apply coreference resolution
    if use_coref:
        ref_corefs = _get_and_replace_coref(references, doc_ids, device)
        if output_dir:
            print(f"===Save coreference outputs to {output_dir}/ref_corefs.pkl===")
            with open(f"{output_dir}/ref_corefs.pkl", 'wb') as f:
                pkl.dump(ref_corefs, f)
        references = [ref_corefs[doc_id][0] for doc_id in doc_ids]

    # get SRL results
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        cuda_device=device)

    ref_srls = {}
    for doc_id, reference in tqdm(zip(doc_ids, references)):
        summary_sents = sent_tokenize(reference.strip())
        srls = []
        for sent in summary_sents:
            sent = sent.strip()
            if sent:
                srl = predictor.predict(sentence=sent)
                srls.append(srl)
        ref_srls[doc_id] = srls
    if output_dir:
        print(f"===Save SRL outputs to {output_dir}/ref_srls.pkl===")
        with open(f"{output_dir}/ref_srls.pkl", 'wb') as f:
            pkl.dump(ref_srls, f)

    return (ref_srls, ref_corefs) if use_coref else (ref_srls,)


def extract_stus(
    references,
    doc_ids=None,
    output_dir=None,
    use_coref=False,
    device=-1
):
    if doc_ids == None:
        doc_ids = [i for i in range(len(references))]

    res = _run_srl(references, doc_ids, output_dir=output_dir, use_coref=use_coref, device=device)

    all_stus = []
    count = 0
    for doc_id in doc_ids:
        stus = _get_srl_list(res[0][doc_id])
        if use_coref:
            stus.extend(res[1][doc_id][1])
        count += len(stus)
        all_stus.append(stus)

    print(f"==={len(all_stus)} references, on average {count / len(all_stus)} STUs per reference===")

    if output_dir:
        print(f"===Save STUs to {output_dir}/STUs.txt===")
        with open(f"{output_dir}/STUs.txt", 'w') as f:
            f.write('\n'.join(['\t'.join(stus) for stus in all_stus]))

    return all_stus


non_terminal_tokens = ['WRB', 'RBR', 'ADVP', 'VBG', '$', "''", 'WHADVP', '-RRB-', 'JJR', 'NAC', 'PRP', 'NNS', 'WP',
                           'VBZ', 'MD', 'WDT', 'NP', 'ADJP', 'PDT', 'EX', 'UH', 'NN', 'NFP', 'SYM', 'PRP$', 'RBS',
                           'FRAG', 'NX', 'CONJP', 'RP', 'WHPP', 'CC', 'VBD', 'LS', '.', 'SBAR', 'TO', 'JJ', 'IN', 'VP',
                           '-LRB-', 'S', 'QP', 'SQ', 'CD', '``', 'X', 'POS', 'XX', 'PP', 'PRT', 'JJS', 'HYPH', ',',
                           'RB', 'VBN', ':', 'VBP', 'DT', 'VB', 'SINV', 'UCP', 'WHNP', 'NNPS', 'NNP']


def _get_features(tokens, tree):
    """
    prepare the feature vector for each reference sentence
    """
    length = len(tokens)
    tree_length = len(tree)
    non_terminal = [0] * len(non_terminal_tokens)
    left, right, depth = 0, 0, 0
    prec = None
    token = ""
    for c in tree:
        if c == " " and token:
            if token in non_terminal_tokens:
                non_terminal[non_terminal_tokens.index(token)] += 1
            prec, token = None, ""
            continue
        if prec == "(":
            token += c
        else:
            prec = c
        if c == "(":
            left += 1
        elif c == ")":
            right += 1
        depth = max(depth, left - right)
    feature = [length, tree_length, depth, length / depth] + non_terminal
    return feature


def mix_scus_stus(
    regressor,
    scus,
    percentage,
    references=None,
    ref_srls_pkl=None,
    ref_corefs=None,
    doc_ids=None,
    output_dir=None,
    use_coref=False,
    device=-1
):
    assert references is not None or ref_srls_pkl is not None, \
        "need to provide either references or the address of ref_srls.pkl"

    if doc_ids == None:
        doc_ids = [i for i in range(len(scus))]

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

    print(f"===Get Easiness Scores===")
    # predict easiness scores
    X, dids, sids = [], [], []
    for doc_id in doc_ids:
        for sent_id, (tokens, tree) in enumerate(all_trees[doc_id]):
            X.append(_get_features(tokens, tree))
            dids.append(doc_id)
            sids.append(sent_id)
    X = xgb.DMatrix(np.array(X))
    bst = xgb.Booster({'nthread': 4})
    bst.load_model(regressor)
    scores = bst.predict(X)
    easiness_dict = {}
    easinesses = []
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
        print(f"===Save STUs_SCUs_percentage{percentage} to {output_dir}/STUs_SCUs_percentage{percentage}.txt===")
        with open(f"{output_dir}/STUs_SCUs_percentage{percentage}.txt", 'w') as f:
            f.write('\n'.join(['\t'.join(us) for us in units]))

    return units


if __name__ == '__main__':
    with open("../data/REALSumm/SCUs.txt", 'r') as f:
        scus = [line.strip().split('\t') for line in f.readlines()]
    with open("../data/REALSumm/ids.txt", 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    mix_scus_stus(regressor="../regressors/TAC08/all_xgb.json",
                  scus=scus, percentage=50,
                  ref_srls_pkl="../data/REALSumm/ref_srls.pkl",
                  ref_corefs="../data/REALSumm/ref_corefs.pkl",
                  output_dir="../data/REALSumm",
                  doc_ids=ids, use_coref=True)