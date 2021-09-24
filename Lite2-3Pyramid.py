import torch
import argparse
from metric import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        type=str, help="Model name: shiyue/roberta-large-tac08 is default.")
    parser.add_argument("--data", default=None,
                        type=str, help="Data name: choose from [nli, tac08, tac09, realsumm, pyrxsum]")
    parser.add_argument("--batch_size", default=32, type=int, help="Eval batch size")
    parser.add_argument("--max_length", default=400, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--unit", required=True, default=None, type=str,
                        help="The file storing SCUs or STUs for references. "
                             "Each line is the SCUs or STUs for one reference. "
                             "Units are separated by '\t'.")
    parser.add_argument("--summary", required=True, default=None, type=str,
                        help="The file storing summaires. Each line is one summary."
                             "Should be aligned with --unit file.")
    parser.add_argument("--weight", default=None, type=str,
                        help="The file storing weights of units."
                             "Should be aligned with --unit file.")
    parser.add_argument("--label", default=None, type=str,
                        help="The file storing gold presence labels of units."
                             "Should be aligned with --unit file.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--detail", action="store_true", help="if output detailed scores for every example")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    with open(args.summary, 'r') as f:
        summaries = [line.strip() for line in f.readlines()]
    with open(args.unit, 'r') as f:
        units = [line.strip().split('\t') for line in f.readlines()]
    labels = None
    if args.label:
        with open(args.label, 'r') as f:
            labels = [[int(label) for label in line.strip().split('\t')] for line in f.readlines()]
    weights = None
    if args.weight:
        with open(args.weight, 'r') as f:
            weights = [[int(weight) for weight in line.strip().split('\t')] for line in f.readlines()]

    res = score(summaries, units, weights=weights, labels=labels, device=device,
                model_type=args.model, data=args.data, batch_size=args.batch_size,
                max_length=args.max_length, cache_dir=args.cache_dir, detail=args.detail)

    print(res)


if __name__ == '__main__':
    main()