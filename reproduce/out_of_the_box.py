import os
import sys
sys.path.insert(0, "..")
from metric import score
from utils import summary_level_correlation, system_level_correlation, get_realsumm_data


def tac08_to_realsumm():
    """
    expected output:
    l3c system-level: 0.9539143281966703, 0.9592307692307691 summary-level: 0.5946730806152126, 0.5776106445576457
    p3c system-level: 0.9438718409760112, 0.9461538461538461 summary-level: 0.6187930427632147, 0.5777794433683304
    l2c system-level: 0.9462351573552207, 0.9461538461538461 summary-level: 0.5903727218444925, 0.568648999313437
    p2c system-level: 0.9445700764541787, 0.9492307692307692 summary-level: 0.6140539736177868, 0.5722168862483084
    """
    units, system_data = get_realsumm_data()

    l3c, p3c, l2c, p2c, human = {}, {}, {}, {}, {}
    for system_name in system_data:
        summaries, labels = system_data[system_name]
        res = score(summaries, units, labels=labels, cache_dir="train/cache", detail=True)

        l3c[system_name] = {i: v for i, v in enumerate(res["l3c"][1])}
        p3c[system_name] = {i: v for i, v in enumerate(res["p3c"][1])}
        l2c[system_name] = {i: v for i, v in enumerate(res["l2c"][1])}
        p2c[system_name] = {i: v for i, v in enumerate(res["p2c"][1])}
        human[system_name] = {i: v for i, v in enumerate(res["human"][1])}

    for metric, prediction in [("l3c", l3c), ("p3c", p3c), ("l2c", l2c), ("p2c", p2c)]:
        sys_pear, sys_spear = system_level_correlation(human, prediction)
        summ_pear, summ_spear = summary_level_correlation(human, prediction)
        print(metric, f"system-level: {sys_pear}, {sys_spear}", f"summary-level: {summ_pear}, {summ_spear}")


if __name__ == '__main__':
    tac08_to_realsumm()
