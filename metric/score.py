import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import data2model


def score(
    summaries,
    units,
    weights=None,
    labels=None,
    data=None,
    model_type=None,
    max_length=400,
    batch_size=32,
    device=-1,
    cache_dir=None,
    detail=False,
):
    assert len(summaries) == len(units), "ERROR: Different number of summaries and units"

    if weights:
        assert len(summaries) == len(weights), "ERROR: Different number of summaries and weights"

    if model_type is None:
        model_type = data2model[data]

    device = "cpu" if device == -1 else f"cuda:{device}"

    model = AutoModelForSequenceClassification.from_pretrained(model_type, cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # prepare data
    input_ids, token_type_ids, attention_mask, ids, wts, lls = [], [], [], [], [], []
    for i, summary in tqdm(enumerate(summaries)):
        for j, unit in enumerate(units[i]):
            tokenized_input_seq_pair = tokenizer.encode_plus(summary, unit,
                                                             max_length=max_length,
                                                             return_token_type_ids=True, truncation=True)
            pad_length = max_length - len(tokenized_input_seq_pair['input_ids'])
            input_ids.append(tokenized_input_seq_pair['input_ids'] + [tokenizer.pad_token_id] * pad_length)
            token_type_ids.append(tokenized_input_seq_pair['token_type_ids'] + [0] * pad_length)
            attention_mask.append(tokenized_input_seq_pair['attention_mask'] + [0] * pad_length)
            ids.append(i)
            wts.append(weights[i][j] if weights else 1)
            lls.append(1 - labels[i][j] if labels else 1)  # entailment 0 = present 1
    input_ids = torch.Tensor(input_ids).long()
    token_type_ids = torch.Tensor(token_type_ids).long()
    attention_mask = torch.Tensor(attention_mask).long()
    wts = torch.Tensor(wts).long()
    lls = torch.Tensor(lls).long()
    ids = torch.Tensor(ids).long()
    eval_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, wts, lls, ids)

    # compute metric scores
    prediction_l3c, prediction_p3c, prediction_p2c, prediction_l2c = {}, {}, {}, {}
    all_weights, gold = {}, {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, weights, lls, ids = batch
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)
            # entailment label is the 0-dim of logits
            logits = torch.stack([outputs[0][:, 0], outputs[0][:, 1] + outputs[0][:, 2]], dim=1)  # 2 classes
            two_labels = torch.argmax(logits, dim=1).reshape(-1)
            two_probs = torch.softmax(logits, dim=1)[:, 0].reshape(-1)  # soft prob
            three_labels = torch.argmax(outputs[0], dim=1).reshape(-1)
            three_probs = torch.softmax(outputs[0], dim=1)[:, 0].reshape(-1)
            for id, thl, thp, twp, twl, input_id, weight, label in zip(ids.cpu().numpy(), three_labels.cpu().numpy(),
                                                                three_probs.cpu().numpy(), two_probs.cpu().numpy(),
                                                                two_labels.cpu().numpy(), input_ids.cpu().numpy(),
                                                                weights.cpu().numpy(), lls.cpu().numpy()):

                if id not in prediction_l3c:
                    prediction_l3c[id] = []
                    prediction_p3c[id] = []
                    prediction_l2c[id] = []
                    prediction_p2c[id] = []
                    gold[id] = []
                    all_weights[id] = []
                prediction_l3c[id].append(1 * weight if thl == 0 else 0)
                prediction_p3c[id].append(thp * weight)
                prediction_l2c[id].append((1 - twl) * weight)
                prediction_p2c[id].append(twp * weight)
                gold[id].append((1 - label) * weight)
                all_weights[id].append(weight)

    l3c, p3c, l2c, p2c = [0] * len(summaries), [0] * len(summaries), [0] * len(summaries), [0] * len(summaries)
    human = [0] * len(summaries)
    for id in prediction_l3c:
        all_weight = sum(all_weights[id])
        l3c[id] = sum(prediction_l3c[id]) / all_weight
        p3c[id] = sum(prediction_p3c[id]) / all_weight
        l2c[id] = sum(prediction_l2c[id]) / all_weight
        p2c[id] = sum(prediction_p2c[id]) / all_weight
        human[id] = sum(gold[id]) / all_weight

    if detail:
        res = {
            "p2c": [np.mean(p2c), p2c],
            "l2c": [np.mean(l2c), l2c],
            "p3c": [np.mean(p3c), p3c],
            "l3c": [np.mean(l3c), l3c],
            "human": [np.mean(human), human] if labels else None,
            "model": model_type
        }
    else:
        res = {
            "p2c": np.mean(p2c),
            "l2c": np.mean(l2c),
            "p3c": np.mean(p3c),
            "l3c": np.mean(l3c),
            "human": np.mean(human) if labels else None,
            "model": model_type
        }
    return res


def human_score(
    labels,
    weights=None,
    detail=False
):
    all_weights, gold = {}, {}
    for id, lls in enumerate(labels):
        gold[id] = []
        all_weights[id] = []
        for j, ll in enumerate(lls):
            weight = weights[id][j] if weights else 1
            gold[id].append(ll * weight)
            all_weights[id].append(weight)
    human = [0] * len(labels)
    for id in gold:
        all_weight = sum(all_weights[id])
        human[id] = sum(gold[id]) / all_weight
    return [np.mean(human), human] if detail else np.mean(human)


if __name__ == '__main__':
    with open("../data/REALSumm/abs_bart_out.summary", 'r') as f:
        summaries = [line.strip() for line in f.readlines()]
    with open("../data/REALSumm/SCUs.txt", 'r') as f:
        units = [line.strip().split('\t') for line in f.readlines()]
    with open("../data/REALSumm/abs_bart_out.label", 'r') as f:
        labels = [[int(label) for label in line.strip().split('\t')] for line in f.readlines()]
    print(score(summaries, units, labels=labels, cache_dir="train/cache"))
    """
    expected output:
    {'p2c': 0.5014365067350771, 'l2c': 0.5159722360972361, 'p3c': 0.43105412152833117, 'l3c': 0.4964144744144744, 'human': 0.48349483849483854}
    """