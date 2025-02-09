import os, argparse, exp_configs, numpy as np
from haven import haven_wizard as hw
from main import get_main_parser, main_func
from abstractive import AbstractiveSummarizer
from extractive import ExtractiveSummarizer


def fetch_datafiles(src_dir, dst_dir, n_train):
    os.makedirs(dst_dir, exist_ok=True)
    for part in ["train", "val", "test"]:
        src_path = os.path.join(src_dir, part + ".source")
        tgt_path = os.path.join(src_dir, part + ".target")
        # select n_train lines randomly
        chosen_line_ids = None
        with open(src_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            if part == "train":
                chosen_line_ids = np.random.choice(
                    range(len(lines)),
                    n_train,
                    replace=False,
                )
            else:
                chosen_line_ids = np.random.choice(
                    range(len(lines)),
                    min(len(lines), 1000),
                    replace=False,
                )

            chosen_lines = [lines[i] for i in chosen_line_ids]
            with open(os.path.join(dst_dir, part + ".source"), "w") as f:
                f.write("\n".join(chosen_lines))

        with open(tgt_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            chosen_lines = [lines[i] for i in chosen_line_ids]
            with open(os.path.join(dst_dir, part + ".target"), "w") as f:
                f.write("\n".join(chosen_lines))


def main(exp_dict, savedir, args):
    dname = exp_dict["dataset"]
    data_dir = os.path.join(savedir, f"tmp_datasets/")
    raw_data_path = os.path.join(data_dir, "raw/")
    processed_data_path = os.path.join(data_dir, "processed/")
    weights_save_path = os.path.join(savedir, "tmp_trained_models")
    exp_dict_path = os.path.join(savedir, "exp_dict.json")

    # prepare data: based on n_train, copy the already prepared .source
    # and .target files into the
    fetch_datafiles(
        src_dir=f"./data/{dname}/transformersum",
        dst_dir=raw_data_path,
        n_train=exp_dict["n_train"],
    )

    # fetch the default arguments for main.py
    main_parser = get_main_parser()
    main_args = main_parser.parse_args([])

    # add some abstractive/extractive specific arguments
    main_args.mode = exp_dict["exp_mode"]
    if main_args.mode == "abstractive":
        main_parser = AbstractiveSummarizer.add_model_specific_args(main_parser)
    else:
        main_parser = ExtractiveSummarizer.add_model_specific_args(main_parser)
    main_args = main_parser.parse_args([])

    # update arguments with desired experiment settings
    main_args.data_path = processed_data_path
    main_args.weights_save_path = weights_save_path
    main_args.max_steps = exp_dict["n_steps"]
    main_args.scoring_function = exp_dict["scoring_function"]
    main_args.savedir = savedir
    main_args.exp_dict_path = exp_dict_path
    main_args.dataset_name = dname
    main_args.relabel = exp_dict["relabel"]
    main_args.n_train = exp_dict["n_train"]
    main_args.do_train = True
    main_args.do_test = True
    main_args.data_type = "txt"
    main_args.learning_rate = exp_dict["lr"]
    main_args.seed = 42 + exp_dict["run#"]

    for curr_cycle in range(exp_dict["n_cycles"]):
        main_args.cycle = curr_cycle
        main_func(main_args)


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
        func=main,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results/chat_summ.ipynb",
        python_binary_path=args.python_binary,
        args=args,
        use_threads=True,
    )
