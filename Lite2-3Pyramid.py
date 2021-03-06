import torch
import argparse
import json
import os
from metric import score, extract_stus, mix_scus_stus


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
    parser.add_argument("--unit", default=None, type=str,
                        help="The file storing SCUs or STUs for references. "
                             "Each line is the SCUs or STUs for one reference. "
                             "Units are separated by '\t'.")
    parser.add_argument("--summary", default=None, type=str,
                        help="The file storing summaires. Each line is one summary."
                             "Should be aligned with --unit file.")
    parser.add_argument("--weight", default=None, type=str,
                        help="The file storing weights of units."
                             "Should be aligned with --unit file.")
    parser.add_argument("--label", default=None, type=str,
                        help="The file storing gold presence labels of units."
                             "Should be aligned with --unit file.")
    parser.add_argument("--detail", action="store_true", help="if output detailed scores for every example")
    parser.add_argument("--extract_stus", action="store_true", help="extract STUs only")
    parser.add_argument("--mix_stus_and_scus", action="store_true", help="mix STUs and SCUs")
    parser.add_argument("--stu_percentage", type=int, default=50, help="The percentage of STUs")
    parser.add_argument("--reference", default=None, type=str,
                        help="The file storing references. Each line is one reference.")
    parser.add_argument("--doc_id", default=None, type=str,
                        help="The file storing document ids. Each line is one document id.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory to save output files from extracting STUs.")
    parser.add_argument("--srl_file", default=None, type=str,
                        help="The intermediate file saving SRL results.")
    parser.add_argument("--coref_file", default=None, type=str,
                        help="The intermediate file saving coreference results.")
    parser.add_argument("--scus_file", default=None, type=str,
                        help="The file saving SCUs.")
    parser.add_argument("--regressor", default=None, type=str,
                        help="The regressor used for mixing STUs and SCUs.")
    parser.add_argument("--use_coref", action="store_true",
                        help="if apply coreference resolution for STU extraction")
    parser.add_argument("--device", type=int, default=0,
                        help="The ID of the GPU to use, -1 if CPU")
    parser.add_argument("--output_file",
                        help="The output file to save the scores to")

    args = parser.parse_args()

    if args.extract_stus:
        assert args.reference is not None, "need to input the file storing references"
        with open(args.reference, 'r') as f:
            references = [line.strip() for line in f.readlines()]
        doc_ids = None
        if args.doc_id:
            with open(args.doc_id, 'r') as f:
                doc_ids = [line.strip() for line in f.readlines()]
        extract_stus(references, doc_ids, output_dir=args.output_dir, use_coref=args.use_coref,
                     device=args.device)
    elif args.mix_stus_and_scus:
        with open(args.scus_file, 'r') as f:
            scus = [line.strip().split('\t') for line in f.readlines()]
        doc_ids = None
        if args.doc_id:
            with open(args.doc_id, 'r') as f:
                doc_ids = [line.strip() for line in f.readlines()]
        if args.srl_file and args.coref_file:
            mix_scus_stus(args.regressor, scus, args.stu_percentage,
                          ref_srls_pkl=args.srl_file, ref_corefs=args.coref_file,
                          doc_ids=doc_ids, output_dir=args.output_dir,
                          use_coref=args.use_coref, device=args.device)
        else:
            assert args.reference is not None, "need to input the file storing references"
            with open(args.reference, 'r') as f:
                references = [line.strip() for line in f.readlines()]
            mix_scus_stus(args.regressor, scus, args.stu_percentage, references=references,
                          doc_ids=doc_ids, output_dir=args.output_dir,
                          use_coref=args.use_coref, device=args.device)
    else:
        assert args.summary is not None, "need to input the file storing summaries"
        assert args.unit is not None, "need to input the file storing units"

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

        res = score(summaries, units, weights=weights, labels=labels, device=args.device,
                    model_type=args.model, data=args.data, batch_size=args.batch_size,
                    max_length=args.max_length, cache_dir=args.cache_dir, detail=args.detail)

        print(res)

        if args.output_file is not None:
            dirname = os.path.dirname(args.output_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(args.output_file, "w") as out:
                out.write(json.dumps(res))


if __name__ == '__main__':
    main()