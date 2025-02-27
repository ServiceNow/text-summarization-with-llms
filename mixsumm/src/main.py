import json, logging, random, time, os, numpy as np, torch, datasets as nlp, warnings, openai
from argparse import ArgumentParser
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin

from abstractive import AbstractiveSummarizer
from extractive import ExtractiveSummarizer
from helpers import StepCheckpointCallback

from haven import haven_utils as hu

warnings.simplefilter("ignore")
logger = logging.getLogger(__name__)



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


def train_test_step(args):
    raw_path = args.data_path.replace("processed", "raw")
    # ^^ ${savedir}/tmp_datasets/raw/

    summarizer = ExtractiveSummarizer
    score_list_savedir = os.path.join(args.savedir, "score_list.pkl")
    if os.path.exists(score_list_savedir):
        score_list = hu.load_pkl(score_list_savedir)
    else:
        score_list = []

    if args.load_weights:
        model = summarizer(hparams=args)
        checkpoint = torch.load(
            args.load_weights, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["state_dict"], strict=args.no_strict)
    elif args.load_from_checkpoint:
        try:
            model = summarizer.load_from_checkpoint(
                args.load_from_checkpoint, strict=args.no_strict
            )
        except RuntimeError as e:
            e_str = str(e)
            if (
                "Missing key(s) in state_dict" in e_str
                or "word_embedding_model.embeddings.position_ids" in e_str
            ):
                print(
                    "The below is a common issue. Due to the `transformers` update "
                    "from 3.0.2 to 3.1.0, models trained in versions <3.0.2 need to be "
                    "loaded with the `--no_strict` argument. More details can be found at "
                    "huggingface/transformers#6882."
                )
            raise e

        # The model is loaded with self.hparams.data_path set to the directory where the data
        # was located during training. When loading the model, it may be desired to change
        # the data path, which the below line accomplishes.
        if args.data_path:
            model.hparams.data_path = os.path.join(args.savedir, args.data_path)
        # Same as above but for `test_use_pyrouge`
        if args.test_use_pyrouge:
            model.hparams.test_use_pyrouge = args.test_use_pyrouge
    else:
        model = summarizer(hparams=args)

    # Create learning rate logger
    lr_logger = LearningRateMonitor()
    args.callbacks = [lr_logger]

    # if args.use_logger == "wandb":
    #     wandb_logger = WandbLogger(
    #         project=args.wandb_project, log_model=(not args.no_wandb_logger_log_model)
    #     )
    #     args.logger = wandb_flogger

    if args.use_custom_checkpoint_callback:
        try:
            args.checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                every_n_epochs=1,
                verbose=True,
                monitor="rouge2-fmeasure",
            )
        except TypeError:
            logger.warning(
                "'every_n_epochs' parameter of ModelCheckpoint is not found. "
                + "Defaulting to its old name, 'period'."
            )
            args.checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                period=1,
                verbose=True,
                monitor="rouge2-fmeasure",
            )

    if args.custom_checkpoint_every_n:
        custom_checkpoint_callback = StepCheckpointCallback(
            step_interval=args.custom_checkpoint_every_n,
            save_path=args.weights_save_path,
        )
        args.callbacks.append(custom_checkpoint_callback)

    if args.plugins and args.plugins.startswith("deepspeed"):
        deepspeed_config_path = args.plugins.split(":")[1]
        with open(deepspeed_config_path, "r") as deepspeed_config_file:
            deepspeed_config = json.load(deepspeed_config_file)
        args.plugins = DeepSpeedPlugin(config=deepspeed_config)

    trainer = Trainer.from_argparse_args(args)

    if args.lr_find:
        lr_finder = trainer.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        logger.info("Recommended Learning Rate: %s", new_lr)

    # remove `args.callbacks` if it exists so it does not get saved with the model
    # (would result in crash)
    if args.custom_checkpoint_every_n:
        del args.callbacks

    if args.do_train:
        trainer.fit(model)
    if args.do_test:
        results = trainer.test(model)
        curr_train_set_size = len(
            open(os.path.join(raw_path, "train.source")).readlines()
        )
        results[0]["cycle"] = args.cycle
        results[0]["train_size"] = curr_train_set_size
        results[0]["dataset"] = args.dataset_name
        results[0]["augmentation_method"] = args.augmentation_method
        results[0]["scoring_function"] = args.n_augment
        results[0]["n_train"] = args.n_train
        score_list.append(results[0])
        hu.save_pkl(score_list_savedir, score_list)


def main(args):
    """Main training + testing code."""
    # remove old files or prepare data
    data_path = args.data_path  # ${savedir}/tmp_datasets/processed/
    raw_path = data_path.replace("processed", "raw")
    # ^^ ${savedir}/tmp_datasets/raw/
    os.system(f"rm {os.path.join(args.data_path, '*.txt')}")
    python_bin = "/mnt/home/miniconda/envs/transformersum/bin/python"
    os.system(
        f"{python_bin} convert_to_extractive.py {raw_path} --base_output_path {data_path} --shard_interval 5000 --add_target_to test --compression"
    )

    if args.seed:
        set_seed(args.seed)

    exp_dict = json.load(open(args.exp_dict_path))
    print(exp_dict)

    train_test_step(args)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # parametrize the network: general options
    parser.add_argument(
        "--mode",
        type=str,
        default="abstractive",
        choices=["extractive", "abstractive"],
        help="Extractive or abstractive summarization training. Default is 'extractive'.",
    )

    parser.add_argument(
        "--dataset_name",
        default="tweetsumm",
        type=str,
        choices=["tweetsumm", "wikihow", "arxiv-pubmed"],
    )

    parser.add_argument(
        "--default_root_dir",
        type=str,
        help="Default path for logs and weights.",
    )
    parser.add_argument(
        "--weights_save_path",
        type=str,
        help="""Where to save weights if specified. Will override `--default_root_dir` for
        checkpoints only. Use this if for whatever reason you need the checkpoints stored in
        a different place than the logs written in `--default_root_dir`.
        If you are using the `wandb` logger, then you must also set `--no_wandb_logger_log_model`
        when using this option. Model weights are saved with the wandb logs to be uploaded to
        wandb.ai by default. Setting this option without setting `--no_wandb_logger_log_model`
        effectively creates two save paths, which may crash the script.""",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,  # 2e-3 and 5e-5 and 3e-6
        type=float,
        help="The initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--min_steps",
        default=None,
        type=int,
        help="Limits training to a minimum number number of steps",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Limits training to a max number number of steps",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="""Accumulates grads every k batches. A single step is one gradient accumulation cycle,
        so setting this value to 2 will cause 2 batches to be processed for each step.""",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        default=1,
        type=int,
        help="Check val every n train epochs.",
    )
    parser.add_argument(
        "--gpus",
        default=-1,
        type=int,
        help="Number of GPUs to train on or Which GPUs to train on. (default: -1 (all gpus))",
    )
    parser.add_argument(
        "--gradient_clip_val", default=1.0, type=float, help="Gradient clipping value"
    )
    parser.add_argument(
        "--overfit_batches",
        default=0.0,
        type=float,
        help="Uses this much data of all datasets (training, validation, test). Useful "
        + "for quickly debugging or trying to overfit on purpose.",
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).",
    )
    parser.add_argument(
        "--limit_train_batches",
        default=1.0,
        type=float,
        help="How much of training dataset to check. Useful when debugging or testing "
        + "something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--limit_val_batches",
        default=1.0,
        type=float,
        help="How much of validation dataset to check. Useful when debugging or testing something "
        + "that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--limit_test_batches",
        default=1.0,
        type=float,
        help="How much of test dataset to check.",
    )
    parser.add_argument(
        "--amp_level",
        type=str,
        default=None,
        help="The optimization level to use (O1, O2, etc…) for 16-bit GPU precision (using "
        + "NVIDIA apex under the hood).",
    )
    parser.add_argument(
        "--n_train",
        default=50,
        type=int,
        help="Number of original labeled training examples to use",
    )
    parser.add_argument(
        "--n_augment",
        default=50,
        type=int,
        help="Number of new training examples to include in each cycle",
    )
    parser.add_argument(
        "--augmentation_method",
        default="eda",
        type=str,
        help="Data augmentation method to use",
    )
    parser.add_argument(
        "--amp_backend",
        type=str,
        default="native",
        choices=["native", "apex"],
        help="PyTorch Lightning amp_backend ('native' or 'apex')",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for reproducible results. Can negatively impact performace in some cases.",
    )
    parser.add_argument(
        "--profiler",
        default=None,
        type=str,
        choices=["simple", "advanced"],
        help="To profile individual steps during training and assist in identifying bottlenecks.",
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        default=50,
        type=int,
        help="How often to refresh progress bar (in steps). In notebooks, faster refresh rates "
        + "(lower number) is known to crash them because of their screen refresh rates, so raise "
        + "it to 50 or more.",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        default=2,
        type=int,
        help="Sanity check runs n batches of val before starting the training routine. This "
        + "catches any bugs in your validation without having to wait for the first "
        + "validation check.",
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        help="How often within one training epoch to check the validation set. Can specify "
        + "as float or int. Use float to check within a training epoch. Use int to check every "
        + "n steps (batches).",
    )
    parser.add_argument(
        "--use_logger",
        default="wandb",
        type=str,
        choices=["tensorboard", "wandb"],
        help="Which program to use for logging. If `wandb` is chosen then model weights "
        + "will automatically be uploaded to wandb.ai.",
    )
    parser.add_argument(
        "--wandb_project",
        default="transformerextsum-private",
        type=str,
        help="The wandb project to save training runs to if `--use_logger` is set to `wandb`.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (save memory at the expense of a slower backward "
        + "pass) for the word embedding model. "
        + "More info: https://github.com/huggingface/transformers/pull/4659#issue-424841871",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        type=str,
        choices=["dp", "ddp", "ddp_cpu", "ddp2"],
        help="The accelerator backend to use (previously known as distributed_backend).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Run the training procedure."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Run the testing procedure."
    )
    parser.add_argument(
        "--load_weights",
        default=False,
        type=str,
        help="""Loads the model weights from a given checkpoint. Hyperparameters are initialized
        from command line arguments. This can be used to change paramters between the training
        and testing stages, for example.""",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        default=False,
        type=str,
        help="Loads the model weights and hyperparameters from a given checkpoint.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="To resume training from a specific checkpoint pass in the path here. Automatically "
        + "restores model, epoch, step, LR schedulers, apex, etc...",
    )
    parser.add_argument(
        "--use_custom_checkpoint_callback",
        action="store_true",
        help="""Use the custom checkpointing callback specified in `main()` by
        `args.checkpoint_callback`. By default this custom callback saves the model every
        epoch and never deletes the saved weights files. You can change the save path by
        setting the `--weights_save_path` option.""",
    )
    parser.add_argument(
        "--custom_checkpoint_every_n",
        type=int,
        default=None,
        help="""The number of steps between additional checkpoints. By default checkpoints are
        saved every epoch. Setting this value will save them every epoch and every N steps. This
        does not use the same callback as `--use_custom_checkpoint_callback` but instead uses a
        different class called `StepCheckpointCallback`. When using this callback, you must specify
        the save path with the `--weights_save_path` option.""",
    )
    parser.add_argument(
        "--no_wandb_logger_log_model",
        action="store_true",
        help="Only applies when using the `wandb` logger. Set this argument to NOT save "
        + "checkpoints in wandb directory to upload to W&B servers.",
    )
    parser.add_argument(
        "--auto_scale_batch_size",
        default=None,
        type=str,
        help="""Auto scaling of batch size may be enabled to find the largest batch size that fits
        into memory. Larger batch size often yields better estimates of gradients, but may also
        result in longer training time. Currently, this feature supports two modes 'power' scaling
        and 'binsearch' scaling. In 'power' scaling, starting from a batch size of 1 keeps doubling
        the batch size until an out-of-memory (OOM) error is encountered. Setting the argument to
        'binsearch' continues to finetune the batch size by performing a binary search. 'binsearch'
        is the recommended option.""",
    )
    parser.add_argument(
        "--lr_find",
        action="store_true",
        help="Runs a learning rate finder algorithm (see https://arxiv.org/abs/1506.01186) "
        + "before any training, to find optimal initial learning rate.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        help="""Which optimizer to use: `adamw` (default), `ranger`, `qhadam`, `radam`, or
        `adabound`.""",
    )
    parser.add_argument(
        "--ranger-k",
        default=6,
        type=int,
        help="""Ranger (LookAhead) optimizer k value (default: 6). LookAhead keeps a single
        extra copy of the weights, then lets the internalized 'faster' optimizer (for Ranger,
        that's RAdam) explore for 5 or 6 batches. The batch interval is specified via the
        k parameter.""",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. Only active if `--use_scheduler` is set to linear.",
    )
    parser.add_argument(
        "--no_strict",
        action="store_false",
        help="Load a model with `strict` mode disabled. This will *not* enforce that the keys "
        + "in `state_dict` match the keys returned by the module's `state_dict()` function.",
    )
    parser.add_argument(
        "--use_scheduler",
        default=False,
        help="""Three options:
        1. `linear`: Use a linear schedule that inceases linearly over `--warmup_steps` to `--learning_rate` then decreases linearly for the rest of the training process.
        2. `onecycle`: Use the one cycle policy with a maximum learning rate of `--learning_rate`.
        (default: False, don't use any scheduler)
        3. `poly`: polynomial learning rate decay from `--learning_rate` to `--end_learning_rate`""",  # noqa: E501
    )
    parser.add_argument(
        "--end_learning_rate",
        default=2e-6,
        type=float,
        help="The ending learning rate when `--use_scheduler` is poly.",
    )
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument(
        "--plugins",
        default=None,
        type=str,
        help="Allows you to connect arbitrary backends. Run `pip install deepspeed mpi4py` to use"
        + "deepspeed plugin.",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    parser.add_argument("--savedir", type=str, help="Directory containing the dataset.")

    parser.add_argument("--exp_dict_path", type=str, help="path to exp_dict.")

    parser.add_argument(
        "--cycle", type=int, default=0, help="Directory containing the dataset."
    )

    main_args = parser.parse_known_args()

    if main_args[0].mode == "abstractive":
        parser = AbstractiveSummarizer.add_model_specific_args(parser)
    else:
        parser = ExtractiveSummarizer.add_model_specific_args(parser)

    if main_args[0].custom_checkpoint_every_n and (not main_args[0].weights_save_path):
        logger.error(
            "You must specify the `--weights_save_path` to use `--custom_checkpoint_every_n`."
        )

    if (
        main_args[0].plugins
        and main_args[0].plugins.startswith("deepspeed")
        and (":" not in main_args[0].plugins)
    ):
        logger.error(
            "If you are using the 'deepspeed' plugin, you must specify the path the to "
            + "deepspeed config like so: `--plugins deepspeed:/path/to/config.json`."
        )

    main_args = parser.parse_args()

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(main_args.logLevel),
    )

    # Set the `nlp` logging verbosity since its default is not INFO.
    # If the verbosity is not set back to the default for the library, an abundance
    # of output will be printed.
    # See https://huggingface.co/docs/datasets/package_reference/logging_methods.html.
    nlp.logging.set_verbosity(nlp.logging.WARNING)

    # trainval(main_args)
    main(main_args)
