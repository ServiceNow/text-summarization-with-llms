# reads the .jsonl formatted tweetsumm dataset
# and converts it into .source and .target format
# as required by this repo

import json, os


def tweetsum_data():
    for part in ["train", "valid", "test"]:
        ds_path = os.path.join("src/data/tweetsumm/", part + ".jsonl")
        with open(ds_path, "r") as f:
            ds = f.readlines()
        ds = [json.loads(line) for line in ds]
        source = [
            line["dialog"].replace("\t", "").replace("\n", " [SEP] ").strip()
            for line in ds
        ]
        target = [
            line["extractive_summaries"][0].replace("\t", "").replace("\n", " ").strip()
            for line in ds
        ]
        with open(
            os.path.join("src/data/tweetsumm/transformersum", part + ".source"), "w"
        ) as f:
            f.write("\n".join(source))
        with open(
            os.path.join("src/data/tweetsumm/transformersum", part + ".target"), "w"
        ) as f:
            f.write("\n".join(target))


def wikihow_arxiv_pubmed_data(dname):
    with open(f"src/data/{dname}/transformersum/train.source") as f:
        src = f.readlines()
    with open(f"src/data/{dname}/transformersum/train.target") as f:
        tgt = f.readlines()

    o = [
        json.dumps({"dialog": src[i].strip(), "abstractive_summary": tgt[i].strip()})
        for i in range(len(src))
    ]
    with open(f"src/data/{dname}/train.jsonl", "w") as f:
        f.write("\n".join(o))


if __name__ == "__main__":
    # tweetsum_data()
    wikihow_arxiv_pubmed_data(dname="arxiv-pubmed")
