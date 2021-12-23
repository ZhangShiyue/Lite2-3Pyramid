import sys
sys.path.insert(0, "..")
import argparse
import numpy as np
from metric import score, human_score
from utils import summary_level_correlation, system_level_correlation, get_realsumm_data


def realsumm_by_examples(version=2, device=-1):
    with open(f"../data/REALSumm/ids.txt", 'r') as f:
        dids = {line.strip(): i for i, line in enumerate(f.readlines())}
    units, system_data = get_realsumm_data(version=version)

    system_level, summary_level = {}, {}
    for fold in range(1, 6):
        with open(f"../data/REALSumm/by_examples/fold{fold}.id", 'r') as f:
            fold_dids = [dids[line.strip()] for line in f.readlines()]

        fold_units = [units[i] for i in fold_dids]
        fold_system_data = {system_name: [[system_data[system_name][0][i] for i in fold_dids],
                                          [system_data[system_name][1][i] for i in fold_dids]]
                            for system_name in system_data}
        l3c, p3c, l2c, p2c, human = {}, {}, {}, {}, {}
        for system_name in system_data:
            summaries, labels = fold_system_data[system_name]
            res = score(summaries, fold_units, labels=labels if version == 2 else None,
                        model_type=f"shiyue/roberta-large-realsumm-by-examples-fold{fold}",
                        detail=True, device=device)

            l3c[system_name] = {i: v for i, v in enumerate(res["l3c"][1])}
            p3c[system_name] = {i: v for i, v in enumerate(res["p3c"][1])}
            l2c[system_name] = {i: v for i, v in enumerate(res["l2c"][1])}
            p2c[system_name] = {i: v for i, v in enumerate(res["p2c"][1])}
            gold = res["human"][1] if version == 2 else human_score(labels, detail=True)[1]
            human[system_name] = {i: v for i, v in enumerate(gold)}

        for metric, prediction in [("l3c", l3c), ("p3c", p3c), ("l2c", l2c), ("p2c", p2c)]:
            sys_pear, sys_spear = system_level_correlation(human, prediction)
            summ_pear, summ_spear = summary_level_correlation(human, prediction)

            if fold == 1:
                system_level[metric] = {"pear": [sys_pear], "spear": [sys_spear]}
                summary_level[metric] = {"pear": [summ_pear], "spear": [summ_spear]}
            else:
                system_level[metric]["pear"].append(sys_pear)
                system_level[metric]["spear"].append(sys_spear)
                summary_level[metric]["pear"].append(summ_pear)
                summary_level[metric]["spear"].append(summ_spear)
    print(f"================ System Level =================")
    for metric in system_level:
        print(metric, np.mean(system_level[metric]["pear"]), np.mean(system_level[metric]["spear"]))
    print(f"================ Summary Level =================")
    for metric in system_level:
        print(metric, np.mean(summary_level[metric]["pear"]), np.mean(summary_level[metric]["spear"]))


def realsumm_by_systems(version=2, device=-1):
    units, system_data = get_realsumm_data(version=version)

    system_level, summary_level = {}, {}
    for fold in range(1, 6):
        with open(f"../data/REALSumm/by_systems/fold{fold}.sys", 'r') as f:
            fold_systems = [line.strip() for line in f.readlines()]

        fold_system_data = {system_name: system_data[system_name]
                            for system_name in system_data if system_name in fold_systems}
        l3c, p3c, l2c, p2c, human = {}, {}, {}, {}, {}
        for system_name in fold_system_data:
            summaries, labels = fold_system_data[system_name]
            res = score(summaries, units, labels=labels if version == 2 else None,
                        model_type=f"shiyue/roberta-large-realsumm-by-systems-fold{fold}",
                        detail=True, device=device)

            l3c[system_name] = {i: v for i, v in enumerate(res["l3c"][1])}
            p3c[system_name] = {i: v for i, v in enumerate(res["p3c"][1])}
            l2c[system_name] = {i: v for i, v in enumerate(res["l2c"][1])}
            p2c[system_name] = {i: v for i, v in enumerate(res["p2c"][1])}
            gold = res["human"][1] if version == 2 else human_score(labels, detail=True)[1]
            human[system_name] = {i: v for i, v in enumerate(gold)}

        for metric, prediction in [("l3c", l3c), ("p3c", p3c), ("l2c", l2c), ("p2c", p2c)]:
            sys_pear, sys_spear = system_level_correlation(human, prediction)
            summ_pear, summ_spear = summary_level_correlation(human, prediction)

            if fold == 1:
                system_level[metric] = {"pear": [sys_pear], "spear": [sys_spear]}
                summary_level[metric] = {"pear": [summ_pear], "spear": [summ_spear]}
            else:
                system_level[metric]["pear"].append(sys_pear)
                system_level[metric]["spear"].append(sys_spear)
                summary_level[metric]["pear"].append(summ_pear)
                summary_level[metric]["spear"].append(summ_spear)
    print(f"================ System Level =================")
    for metric in system_level:
        print(metric, np.mean(system_level[metric]["pear"]), np.mean(system_level[metric]["spear"]))
    print(f"================ Summary Level =================")
    for metric in system_level:
        print(metric, np.mean(summary_level[metric]["pear"]), np.mean(summary_level[metric]["spear"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="realsumm",
                        type=str, help="Data name: choose from [nli, tac08, tac09, realsumm, pyrxsum]")
    parser.add_argument("--split", default="examples",
                        type=str, help="Split by: examples or systems")
    parser.add_argument("--version", default=2, type=float, help="Lite[version]Pyramid")
    parser.add_argument("--device", type=int, default=-1,
                        help="The ID of the GPU to use, -1 if CPU")

    args = parser.parse_args()

    if args.data == "realsumm":
        if args.split == "examples":
            realsumm_by_examples(args.version, args.device)
        elif args.split == "systems":
            realsumm_by_systems(args.version, args.device)
        else:
            print("invalid split!")