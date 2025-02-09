import numpy as np, torch, random, warnings, logging, os, pickle, exp_configs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import argparse
import os
import pickle
from typing import Dict, Any
from haven import haven_wizard as hw

warnings.simplefilter("ignore")
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> Dict[str, Dataset]:
    datasets = {}
    for split in ["train", "valid", "test"]:
        if split == "train":
            # combine files from org.source and aug.source
            # with open(os.path.join(data_dir, "org.source"), "r") as f:
            #     org_source = f.readlines()
            # with open(os.path.join(data_dir, "aug.source"), "r") as f:
            #     aug_source = f.readlines()
            # documents = org_source + aug_source
            # with open(os.path.join(data_dir, "org.target_abs"), "r") as f:
            #     org_target = f.readlines()
            # with open(os.path.join(data_dir, "aug.target_abs"), "r") as f:
            #     aug_target = f.readlines()
            # summaries = org_target + aug_target
            with open(os.path.join(data_dir, f"{split}.source"), "r") as f:
                documents = f.readlines()
            with open(os.path.join(data_dir, f"{split}.target_abs"), "r") as f:
                summaries = f.readlines()
        else:
            with open(os.path.join(data_dir, f"{split}.source"), "r") as f:
                documents = f.readlines()
            with open(os.path.join(data_dir, f"{split}.target_abs"), "r") as f:
                summaries = f.readlines()
        datasets[split] = Dataset.from_dict(
            {"document": documents, "summary": summaries}
        )
    return datasets


def preprocess_function(examples, tokenizer):
    inputs = examples["document"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.savedir = kwargs.pop("savedir")
        super().__init__(*args, **kwargs)
        self.score_list = []

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self.score_list.append(
                {
                    "n_epochs": epoch,
                    **{
                        k: v
                        for k, v in metrics.items()
                        if k in ["eval_rouge1", "eval_rouge2", "eval_rougeL"]
                    },
                }
            )

            if epoch % 5 == 0 or epoch == int(self.args.num_train_epochs):
                with open("score_list.pkl", "wb") as f:
                    pickle.dump(self.score_list, f)

        return super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
        )


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


def main(exp_dict, savedir: str, args: str):
    set_seed(42)
    dataset_name = exp_dict["dataset"]
    augmentation_type = exp_dict["augmentation_method"]
    # Load the DistilBART model and tokenizer
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load your dataset
    data_dir = os.path.join("./data", dataset_name, "transformersum")
    # data_dir = os.path.join("./data", dataset_name, "transformersum", augmentation_type)
    datasets = load_data(data_dir)

    # Tokenize the dataset
    tokenized_datasets = {
        split: dataset.map(
            lambda examples: preprocess_function(examples, tokenizer), batched=True
        )
        for split, dataset in datasets.items()
    }

    # Define the metrics
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}

        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }

    # Set up the trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results_{dataset_name}_{augmentation_type}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=40,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        savedir=savedir,  # Pass the savedir to CustomTrainer
    )

    # Fine-tune the model
    trainer.train()

    # predict and evaluate on the test set
    predictions = trainer.predict(tokenized_datasets["test"])
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    print(metrics)
    # decode the predictions
    predictions = tokenizer.batch_decode(
        predictions.predictions, skip_special_tokens=True
    )
    # save the predictions as eda.predicted
    with open(
        f"./data/{dataset_name}/transformersum/{augmentation_type}/test.predicted_abs",
        "w",
    ) as f:
        f.write("\n".join(predictions))
        print(
            f"Saved predictions to ./data/{dataset_name}/transformersum/{augmentation_type}/{augmentation_type}.predicted"
        )

    # Save the fine-tuned model
    model.save_pretrained(f"./fine_tuned_distilbart_{dataset_name}_{augmentation_type}")
    tokenizer.save_pretrained(
        f"./fine_tuned_distilbart_{dataset_name}_{augmentation_type}"
    )


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
        results_fname="results/chat_summ_abs.ipynb",
        python_binary_path=args.python_binary,
        python_file_path=f"-m src.abstractive_trainval",
        args=args,
        use_threads=True,
    )
