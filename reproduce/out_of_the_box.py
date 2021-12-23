import os
import sys
sys.path.insert(0, "..")
import argparse
from metric import score, human_score
from utils import summary_level_correlation, system_level_correlation, get_realsumm_data


def tac08_to_realsumm(version=2, device=-1):
    units, system_data = get_realsumm_data(version=version)

    l3c, p3c, l2c, p2c, human = {}, {}, {}, {}, {}
    for system_name in system_data:
        summaries, labels = system_data[system_name]
        res = score(summaries, units, labels=labels if version == 2 else None,
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
        print(metric, f"system-level: {sys_pear}, {sys_spear}", f"summary-level: {summ_pear}, {summ_spear}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tac08",
                        type=str, help="Source data name: choose from [tac08, tac08+tac09, tac08+tac09+realsumm]")
    parser.add_argument("--target", default="realsumm",
                        type=str, help="Target data name: choose from [tac09, realsumm, pyrxsum]")
    parser.add_argument("--version", default=2, type=float, help="Lite[version]Pyramid")
    parser.add_argument("--device", type=int, default=-1,
                        help="The ID of the GPU to use, -1 if CPU")

    args = parser.parse_args()

    if args.source == "tac08" and args.target == "realsumm":
        tac08_to_realsumm(version=args.version, device=args.device)
