import os, argparse, numpy as np, random, torch, logging, warnings, exp_configs
from haven import haven_wizard as hw
from src.augment_utils import do_eda_and_save, do_mixsum_and_save, do_nomixup_and_save

warnings.simplefilter("ignore")
logger = logging.getLogger(__name__)


def fetch_datafiles(src_dir, dst_dir, n_train, n_cycle, n_augment=5):
    # os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_dir, str(n_cycle)), exist_ok=True)
    # do the training data fetching
    aug_src = [l.strip() for l in open(os.path.join(src_dir, "aug.source")).readlines()]
    aug_tgt = [l.strip() for l in open(os.path.join(src_dir, "aug.target")).readlines()]
    org_src = [l.strip() for l in open(os.path.join(src_dir, "org.source")).readlines()]
    org_tgt = [l.strip() for l in open(os.path.join(src_dir, "org.target")).readlines()]

    if n_cycle == 0:
        # select n_train lines randomly
        chosen_line_ids = np.random.choice(range(len(org_src)), n_train, replace=False)
        train_src = [org_src[i].strip() for i in chosen_line_ids]
        train_tgt = [org_tgt[i].strip() for i in chosen_line_ids]
    else:
        # load existing data from the previous cycle and add new EDA examples
        with open(os.path.join(dst_dir, f"{n_cycle-1}/train.source"), "r") as f:
            train_src = f.readlines()
            train_src = [line.strip() for line in train_src]
        with open(os.path.join(dst_dir, f"{n_cycle-1}/train.target"), "r") as f:
            train_tgt = f.readlines()
            train_tgt = [line.strip() for line in train_tgt]
        # add new data to existing data
        # minimum val of n_cycle is 1 (as n_cycle = 0 is handled above)
        aug_start = n_cycle - 1 * n_augment
        train_src.extend(aug_src[aug_start : aug_start + n_augment])
        train_tgt.extend(aug_tgt[aug_start : aug_start + n_augment])

    with open(os.path.join(dst_dir, f"{n_cycle}/train.source"), "w") as f:
        f.write("\n".join(train_src))
    with open(os.path.join(dst_dir, f"{n_cycle}/train.target"), "w") as f:
        f.write("\n".join(train_tgt))

    # do validation and test set fetching
    for part in ["val", "test"]:
        os.system(f"cp {src_dir}/{part}.source {dst_dir}/{n_cycle}/{part}.source")
        os.system(f"cp {src_dir}/{part}.target {dst_dir}/{n_cycle}/{part}.target")

    print("Finished fetching training+val+test datafiles.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning(
        "Deterministic mode can have a performance impact, depending on your model. This means "
        + "that due to the deterministic nature of the model, the processing speed (i.e. "
        + "processed batch items per second) can be lower than when the model is "
        + "non-deterministic."
    )


def trainval(exp_dict, savedir, args):
    set_seed(42)
    dname = exp_dict["dataset"]
    data_dir = os.path.join(savedir, f"tmp_datasets/")
    raw_data_path = os.path.join(data_dir, "raw/")
    processed_data_path = os.path.join(data_dir, "processed/")
    weights_save_path = os.path.join(savedir, "tmp_trained_models")
    exp_dict_path = os.path.join(savedir, "exp_dict.json")

    augmentation_method = exp_dict["augmentation_method"]
    if not os.path.exists(
        f"./data/{dname}/transformersum/{augmentation_method}/aug.source"
    ):
        if augmentation_method == "eda":
            # applies EDA on the original dataset and saves them inside an 'eda' folder
            do_eda_and_save(dname, n_gen=exp_dict["n_augment"])
        elif augmentation_method == "no_mixup":
            # applies LLM augmentations
            do_nomixup_and_save(
                dname,
                n_gen=exp_dict["n_augment"],
                n_train=exp_dict["n_train"],
            )
        elif augmentation_method == "mixsumm":
            # applies MixSumm augmentations
            do_mixsum_and_save(
                dname,
                mode="smart",
                n_gen=exp_dict["n_augment"],
                n_train=exp_dict["n_train"],
            )
        else:
            do_mixsum_and_save(
                dname,
                mode="naive",
                n_gen=exp_dict["n_augment"],
                n_train=exp_dict["n_train"],
            )

    # define python binary
    python_bin = args.python_binary
    for curr_cycle in range(exp_dict["n_cycles"]):
        # create data files to train in each cycle inside dst_dir/curr_cycle
        fetch_datafiles(
            src_dir=f"./data/{dname}/transformersum/{augmentation_method}",
            dst_dir=raw_data_path,
            n_train=exp_dict["n_train"],
            n_cycle=curr_cycle,
            n_augment=exp_dict["n_augment"],
        )
        os.chdir("src")
        command = (
            f"{python_bin} main.py"
            + f" --data_path {os.path.join(processed_data_path, str(curr_cycle))}"
            + f" --weights_save_path {weights_save_path}"
            + f" --max_steps {exp_dict['n_steps']}"
            + f" --savedir {savedir}"
            + f" --exp_dict_path {exp_dict_path}"
            + f" --cycle {curr_cycle}"
            + f" --dataset_name {dname}"
            + f" --n_augment {exp_dict['n_augment']}"
            + f" --augmentation_method {exp_dict['augmentation_method']}"
            + f" --n_train {exp_dict['n_train']}"
            + " --do_train"
            + " --do_test"
            + " --data_type txt"
        )
        os.system(command)
        os.chdir("..")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument(
        "-j", "--job_scheduler", default=None, help="Choose Job Scheduler."
    )
    parser.add_argument(
        "-p", "--python_binary", default="python", help="path to your python executable"
    )

    args, others = parser.parse_known_args()
    print(args)
    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler in ["1", "toolkit"]:
        import job_configs

        job_config = job_configs.JOB_CONFIG

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results/chat_summ.ipynb",
        python_binary_path=args.python_binary,
        python_file_path=f"-m src.trainval",
        args=args,
        use_threads=True,
    )
