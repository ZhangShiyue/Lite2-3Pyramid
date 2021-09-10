import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def evaluate(args):
    with open(args.summaries, 'r') as f:
        summaries = f.readlines()
    with open(args.units, 'r') as f:
        all_units = f.readlines()

    try:
        assert len(summaries) == len(all_units)
    except:
        print("ERROR: --units file and --summaries file should have the same number of lines!")
        exit()

    model = AutoModelForSequenceClassification.from_pretrained(args.model, cache_dir=args.cache_dir).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # prepare data
    input_ids, token_type_ids, attention_mask, id, sys = [], [], [], [], []
    for i, summary in tqdm(enumerate(summaries)):
        summary = summary.strip()
        units = all_units[i].strip().split('\t')
        for unit in units:
            tokenized_input_seq_pair = tokenizer.encode_plus(summary, unit,
                                                             max_length=args.max_length,
                                                             return_token_type_ids=True, truncation=True)
            pad_length = args.max_length - len(tokenized_input_seq_pair['input_ids'])
            input_ids.append(tokenized_input_seq_pair['input_ids'] + [tokenizer.pad_token_id] * pad_length)
            token_type_ids.append(tokenized_input_seq_pair['token_type_ids'] + [0] * pad_length)
            attention_mask.append(tokenized_input_seq_pair['attention_mask'] + [0] * pad_length)
            id.append(i)
    input_ids = torch.Tensor(input_ids).long()
    token_type_ids = torch.Tensor(token_type_ids).long()
    attention_mask = torch.Tensor(attention_mask).long()
    ids = torch.Tensor(id).long()
    print("input size:", input_ids.size())
    eval_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, ids)

    # compute metric scores
    prediction_l3c, prediction_p3c, prediction_p2c, prediction_l2c = [0]*len(summaries), [0]*len(summaries), \
                                                                     [0]*len(summaries), [0]*len(summaries)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, token_type_ids, attention_mask, ids = batch
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
            for id, thl, thp, twp, twl, input_id in zip(ids.cpu().numpy(), three_labels.cpu().numpy(),
                                                        three_probs.cpu().numpy(), two_probs.cpu().numpy(),
                                                        two_labels.cpu().numpy(), input_ids.cpu().numpy()):
                prediction_l3c[id] = 1 if thl == 0 else 0
                prediction_p3c[id] = thp
                prediction_l2c[id] = 1 - twl
                prediction_p2c[id] = twp

    res = {
        "l3c": [np.mean(prediction_l3c), prediction_l3c],
        "p3c": [np.mean(prediction_p3c), prediction_p3c],
        "l2c": [np.mean(prediction_l2c), prediction_l2c],
        "p2c": [np.mean(prediction_p2c), prediction_p2c],
    }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="shiyue/roberta-large-tac08",
                        type=str, help="Model name")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Eval batch size")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--max_length", default=400, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--units", required=True, default=None, type=str,
                        help="The file storing SCUs or STUs for references. "
                             "Each line is the SCUs or STUs for one reference. SCUs or STUs are separated by '\t'.")
    parser.add_argument("--summaries", required=True, default=None, type=str,
                        help="The file storing summaires. Each line is one summary."
                             "Should be aligned with --units file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    return evaluate(args)


if __name__ == '__main__':
    res = main()
    print({"p2c": res['p2c'][0], "l2c": res['l2c'][0], "p3c": res['p3c'][0], "l3c": res['l3c'][0]})