# EXP_GROUPS = summarize.EXP_GROUPS
from haven import haven_utils as hu

EXP_GROUPS = {}
SUMMARY_SIZE = {"tweetsumm": 4, "wikihow": 8, "arxiv-pubmed": 8}


def get_base_exps(dname):
    return hu.cartesian_exp_group(
        {
            "exp_mode": ["abstractive"],  # either 'extractive', 'abstractive'
            # "exp_mode": ["extractive"],  # either 'extractive', 'abstractive'
            "dataset": [dname],
            "model": {
                "name": "mixsumm",
                "backbone": "presumm",
            },
            "n_train": [50],
            "run#": list(range(1)),
            "n_steps": [100],
            "n_cycles": [100],
            "n_augment": [10],
            "run#": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "augmentation_method": [
                # "eda",
                # "mixsumm",
                # "naive_mixsum",
                "no_mixup",
            ],  # eda, mixsum, "naive_mixsum", "no_mixup"
            "summary_size": [SUMMARY_SIZE[dname]],  # number of sentences in summary
        },
        remove_none=True,
    )


EXP_GROUPS["mixsumm"] = []

for dname in ["tweetsumm"]:
    EXP_GROUPS["mixsumm"] += get_base_exps(dname)
