import pickle as pkl
from tqdm import tqdm
from nltk import sent_tokenize
from allennlp.predictors.predictor import Predictor


def get_reference_corefs():
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
    with open("references.txt", 'r') as f:
        references = f.readlines()
    with open("ids.txt", 'r') as f:
        dids = f.readlines()

    all_corefs = {}
    for i, did in enumerate(tqdm(dids)):
        summary = references[i].replace('<t>', ' ').replace('</t>', ' ').strip()
        coref = predictor.predict(document=summary)
        all_corefs[did] = coref
    with open("ref_corefs.pkl", 'wb') as f:
        pkl.dump(all_corefs, f)


def replace_coref():
    with open("ref_corefs.pkl", 'rb') as f:
        all_corefs = pkl.load(f)
    ref_corefs_replaced = {}
    for did in all_corefs:
        words = all_corefs[did]["document"]
        clusters = all_corefs[did]["clusters"]
        all_pronouns = {}
        scus = []
        for cluster in clusters:
            pronouns, others = [], []
            for s, e in cluster:
                span = ' '.join(words[s:e+1]).strip()
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
                    if other.lower() != replacement.lower() and other.lower() not in unique_others\
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
    with open("ref_corefs_replaced.pkl", 'wb') as f:
        pkl.dump(ref_corefs_replaced, f)


def get_reference_srls_coref():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    with open("ref_corefs_replaced.pkl", 'rb') as f:
        data = pkl.load(f)

    all_srls = {}
    for did in tqdm(data):
        summary, _ = data[did]
        summary_sents = sent_tokenize(summary.strip())
        srls = []
        for sent in summary_sents:
            sent = sent.strip()
            if sent:
                srl = predictor.predict(sentence=sent)
                srls.append(srl)
        all_srls[did] = srls
    with open("ref_corefs_replaced_srls.pkl", 'wb') as f:
        pkl.dump(all_srls, f)


def _get_srl_list(sents):
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
                        if position == "B":
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


def get_stus():
    with open("ref_corefs_replaced.pkl", 'rb') as f:
        ref_corefs = pkl.load(f)
    with open("ref_corefs_replaced_srls.pkl", 'rb') as f:
        ref_srls = pkl.load(f)

    with open("ids.txt", 'r') as f:
        dids = f.readlines()

    res = []
    for did in dids:
        stus = _get_srl_list(ref_srls[did]) + ref_corefs[did][1]
        res.append('\t'.join(stus))

    with open("STUs.txt", 'w') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':
    get_reference_corefs()
    replace_coref()
    get_reference_srls_coref()
    get_stus()