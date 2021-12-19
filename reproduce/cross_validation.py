import sys
sys.path.insert(0, "..")
import argparse
import numpy as np
from metric import score, human_score
from utils import summary_level_correlation, system_level_correlation, get_realsumm_data


def realsumm_by_examples(version=2, device=-1):
    """
    version=2 expected output:
    ================ System Level =================
    l3c 0.8417933546081342 0.8010769230769231
    p3c 0.8367324204640783 0.7655384615384616
    l2c 0.873318115166315 0.8461538461538461
    p2c 0.88526218266187 0.8573846153846155
    ================ Summary Level =================
    l3c 0.5649634208524078 0.5384053984871865
    p3c 0.6379487042335673 0.5964372547536069
    l2c 0.5686841213142527 0.542580658037932
    p2c 0.6421714799110585 0.6005035335930164

    version=2.5 expected output:
    ================ System Level =================
    l3c 0.8294911596118257 0.7876923076923077
    p3c 0.8684428603781118 0.8172307692307692
    l2c 0.8635053811630928 0.8318461538461538
    p2c 0.9003803036017544 0.8859999999999999
    ================ Summary Level =================
    l3c 0.5150350085079959 0.4921186195829096
    p3c 0.6129364373373607 0.5708284376254478
    l2c 0.5247998339992539 0.50582631573561
    p2c 0.6156295310098946 0.5715489127292626

    version=3 expected output:
    ================ System Level =================
    l3c 0.8125769965944079 0.7795384615384615
    p3c 0.8841078645351625 0.8683076923076924
    l2c 0.8502911457458968 0.8118461538461539
    p2c 0.8857783322303019 0.8681538461538463
    ================ Summary Level =================
    l3c 0.4625040999441442 0.4387146552055542
    p3c 0.5802680664600168 0.5409421497592383
    l2c 0.4796741509889696 0.4585034329414799
    p2c 0.5746644786840002 0.5324422041594156
    """
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
    """
    version=2 expected output:
    ================ System Level =================
    l3c 0.7502881735928073 0.6799999999999999
    p3c 0.7514473158330084 0.7
    l2c 0.8136240787614536 0.7799999999999999
    p2c 0.7389623756730259 0.7199999999999999
    ================ Summary Level =================
    l3c 0.5099953721863157 0.49821640182533294
    p3c 0.5477781945069251 0.5175346450835481
    l2c 0.5339576668182527 0.5171383697651195
    p2c 0.5529397909320518 0.5206720621937355

    version=3 expected output:
    ================ System Level =================
    l3c 0.7715564782870926 0.74
    p3c 0.7774456941910397 0.76
    l2c 0.8235133749974825 0.8399999999999999
    p2c 0.7804174278938876 0.7599999999999999
    ================ Summary Level =================
    l3c 0.43228962704258034 0.42105945909959974
    p3c 0.4879742684037576 0.4595304316927783
    l2c 0.45217790754278975 0.43911306320979093
    p2c 0.48765488022472103 0.46381178796934996

    TODO summary level results are slightly different from old numbers
    """
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