from collections import defaultdict

data2model = defaultdict(lambda: "shiyue/roberta-large-tac08")
data2model.update(
    {
        "nli": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "tac08": "shiyue/roberta-large-tac08",
        "tac09": "shiyue/roberta-large-tac09",
        "realsumm": "shiyue/roberta-large-realsumm",
        "pyrxsum": "shiyue/roberta-large-pyrxsum",
    }
)