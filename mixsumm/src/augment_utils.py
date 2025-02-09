import json, numpy as np, os
from tqdm import tqdm
from src.eda_utils import eda
from src.llm_utils import mixsum, llm_relabelling, ensure_extractiveness, no_mixup


def save_aug(
    dataset_name,
    augmentation_method,
    aug_source,
    aug_target,
    org_source,
    org_target,
    chosen_summary_ids=None,
):
    # save augmented and original data
    with open(
        f"./data/{dataset_name}/transformersum/{augmentation_method}/aug.source", "w"
    ) as f:
        f.write("\n".join(aug_source))
    with open(
        f"./data/{dataset_name}/transformersum/{augmentation_method}/aug.target", "w"
    ) as f:
        f.write("\n".join(aug_target))
    with open(
        f"./data/{dataset_name}/transformersum/{augmentation_method}/org.source", "w"
    ) as f:
        f.write("\n".join(org_source))
    with open(
        f"./data/{dataset_name}/transformersum/{augmentation_method}/org.target", "w"
    ) as f:
        f.write("\n".join(org_target))
    if chosen_summary_ids:
        with open(
            f"./data/{dataset_name}/transformersum/{augmentation_method}/chosen_summary_ids.npy",
            "wb",
        ) as f:
            np.save(f, chosen_summary_ids)


def read_training_set(dataset_name):
    data_path = f"./data/{dataset_name}/train.jsonl"
    # read jsonl file
    with open(data_path, "r") as f:
        training_set = f.readlines()
        training_set = [json.loads(l) for l in training_set]
    return training_set


def do_eda_and_save(dataset_name, n_gen=1, n_train=50):
    """
    Applies Easy Data Augmentation method on the training set to generate new training examples.
    """
    os.makedirs(f"./data/{dataset_name}/transformersum/eda", exist_ok=True)
    training_set = read_training_set(dataset_name)[:n_train]
    aug_source, org_source, aug_target, org_target = [], [], [], []
    chosen_summary_ids = []
    for data in tqdm(training_set, desc="Applying EDA"):
        dialog_formatted = data["dialog_formatted"]
        dialog_formatted = dialog_formatted.splitlines()
        chosen_summary_id = np.random.choice(
            range(len(data["extractive_line_numbers"]))
        )
        chosen_summary_ids.append(chosen_summary_id)
        extractive_line_numbers = data["extractive_line_numbers"][chosen_summary_id]

        for i in range(n_gen):
            if i == 0:
                org_source.append(" [SEP] ".join(dialog_formatted))
                org_target.append(
                    data["extractive_summaries"][chosen_summary_id]
                    .replace("\n", " [SEP] ")
                    .replace("  ", " ")
                )
            modified_dialog = []
            for _, line in enumerate(dialog_formatted):
                # apply EDA
                customer = "Customer:" in line
                if customer:
                    line = line.replace("Customer:", "")
                else:
                    line = line.replace("Agent:", "")
                line = line.strip()
                if line == "":
                    print("Empty line")
                    continue
                line = eda(line, num_aug=1)[0]
                line = "Customer:" + line if customer else "Agent:" + line
                modified_dialog.append(line)

            new_summary = " [SEP] ".join(
                [modified_dialog[i] for i in extractive_line_numbers]
            )
            modified_dialog = " [SEP] ".join(modified_dialog)
            aug_source.append(modified_dialog)
            aug_target.append(new_summary)

    save_aug(
        dataset_name,
        "eda",
        aug_source,
        aug_target,
        org_source,
        org_target,
        chosen_summary_ids,
    )


def do_nomixup_and_save(dataset_name, n_gen=1, n_train=50):
    os.makedirs(f"./data/{dataset_name}/transformersum/mixsum", exist_ok=True)
    aug_source, aug_target = [], []
    # use org_source and org_target from when we did EDA
    org_source = [
        l.strip()
        for l in open(
            f"./data/{dataset_name}/transformersum/eda/org.source"
        ).readlines()
    ]
    org_target = [
        l.strip()
        for l in open(
            f"./data/{dataset_name}/transformersum/eda/org.target"
        ).readlines()
    ]
    # n_train is the number of labeled examples available
    org_source = org_source[:n_train]
    for idx, dialog_formatted in tqdm(
        enumerate(org_source),
        desc=f"Applying no mixup Augmentation",
        total=len(org_source),
    ):
        ids = list(range(len(org_source)))
        ids.pop(idx)
        # apply LLM perturbation to the dialog
        new_dialogs = no_mixup(dialog_formatted, n_gen=n_gen)
        assert len(new_dialogs) == n_gen
        for new_dialog in new_dialogs:
            # generate a new summary using LLM
            new_summary = ensure_extractiveness(llm_relabelling(new_dialog), new_dialog)
            new_summary = new_summary.replace("\n", " [SEP] ").replace("  ", " ")
            new_dialog = new_dialog.replace("\n", " [SEP] ").replace("  ", " ")
            aug_source.append(new_dialog)
            aug_target.append(new_summary)

    save_aug(
        dataset_name,
        "no_mixup",
        aug_source,
        aug_target,
        org_source,
        org_target,
        chosen_summary_ids=None,
    )


def do_mixsum_and_save(dataset_name, mode="naive", n_gen=1, n_train=50):
    os.makedirs(f"./data/{dataset_name}/transformersum/mixsum", exist_ok=True)
    aug_source, aug_target = [], []
    # use org_source and org_target from when we did EDA
    org_source = [
        l.strip()
        for l in open(
            f"./data/{dataset_name}/transformersum/eda/org.source"
        ).readlines()
    ]
    org_target = [
        l.strip()
        for l in open(
            f"./data/{dataset_name}/transformersum/eda/org.target"
        ).readlines()
    ]
    # n_train is the number of labeled examples available
    org_source = org_source[:n_train]
    for idx, dialog_formatted in tqdm(
        enumerate(org_source),
        desc=f"Applying {mode} MixSum Augmentation",
        total=len(org_source),
    ):
        ids = list(range(len(org_source)))
        ids.pop(idx)
        other_conv_id = np.random.choice(ids)
        # apply LLM perturbation to the dialog
        new_dialogs = mixsum(
            dialog_formatted,
            org_source[other_conv_id],
            mode=mode,
            n_gen=n_gen,
        )
        assert len(new_dialogs) == n_gen
        for new_dialog in new_dialogs:
            # generate a new summary using LLM
            new_summary = ensure_extractiveness(llm_relabelling(new_dialog), new_dialog)
            new_summary = new_summary.replace("\n", " [SEP] ").replace("  ", " ")
            new_dialog = new_dialog.replace("\n", " [SEP] ").replace("  ", " ")
            aug_source.append(new_dialog)
            aug_target.append(new_summary)

    save_aug(
        dataset_name,
        "naive_mixsum" if mode == "naive" else "mixsumm",
        aug_source,
        aug_target,
        org_source,
        org_target,
        chosen_summary_ids=None,
    )
