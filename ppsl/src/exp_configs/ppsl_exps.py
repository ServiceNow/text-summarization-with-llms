# EXP_GROUPS = summarize.EXP_GROUPS
from haven import haven_utils as hu

EXP_GROUPS = {}
SUMMARY_SIZE = {"tweetsumm": 4, "wikihow": 8, "arxiv-pubmed": 8}


def get_base_exps(dname):
    return hu.cartesian_exp_group(
        {
            "exp_mode": ["extractive"],  # either 'extractive', 'abstractive'
            "dataset": [dname],
            "model": {
                "name": "ssl",
                "backbone": "presumm",  # llm
            },
            "n_train": [1, 50, 100, 500],
            "run#": list(range(3)),
            "n_steps": [1, 100, 1000],
            "n_cycles": [50],
            "scoring_function": ["presumm_llm"],  # 'presumm', 'presumm_llm'
            "relabel": [True],  # relabel using LLM
            "summary_size": [SUMMARY_SIZE[dname]],  # number of sentences in summary
            "gen_engine": ["llama-3-70b"],  # "llama-3-70b"
            "lr": [2e-5],  # 2e-3, 5e-5, 3e-6
        },
        remove_none=True,
    )


EXP_GROUPS["ssl"] = []

for dname in ["tweetsumm", "wikihow", "arxiv-pubmed"]:
    EXP_GROUPS["ssl"] += get_base_exps(dname)
