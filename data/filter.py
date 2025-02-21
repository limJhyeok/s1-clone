from tqdm import tqdm
from dotenv import load_dotenv
import os
from datasets import load_dataset, Dataset
import utils
import logging
import re

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")

logging.basicConfig(level=logging.INFO)


def drop_na(dataset: Dataset, featurized_dict: dict):
    no_na = set()
    for example in tqdm(dataset):
        qhash = utils.question_hash(example["question"])
        if qhash in featurized_dict:
            features = featurized_dict[qhash]
            no_nans = all([v is not None for v in features.values()])
            if no_nans:
                no_na.add(qhash)
    dataset = dataset.filter(lambda x: utils.question_hash(x["question"]) in no_na)
    return dataset


def drop_weird_string_patterns(dataset: Dataset):
    no_weird_string = set()
    for example in tqdm(dataset):
        qhash = utils.question_hash(example["question"])
        question = example["question"]
        if (
            "[asy]" not in question
            and "![Image]" not in question
            and not any(f"p{i}." in question for i in range(1, 1000))
            and not re.search(r"\*\*(?=.*\d)[^\s\]]*\*\*", question)
        ):
            no_weird_string.add(qhash)
        elif "[asy]" in question:
            if "AIME" in example["source_type"]:
                no_weird_string.add(qhash)

    dataset = dataset.filter(
        lambda x: utils.question_hash(x["question"]) in no_weird_string
    )
    print("Length of dataset: ", len(dataset))

    return dataset


def drop_qwen_correct(dataset: Dataset, featurized_dict: dict):
    no_qwen_correct = set()
    for example in tqdm(dataset):
        qhash = utils.question_hash(example["question"])
        features = featurized_dict[qhash]
        if not (features["isqwen32bcorrect"] or features["isqwen7bcorrect"]):
            no_qwen_correct.add(qhash)
    dataset = dataset.filter(
        lambda x: utils.question_hash(x["question"]) in no_qwen_correct
    )
    print("Length of dataset: ", len(dataset))
    return dataset


if __name__ == "__main__":
    if HF_USERNAME:
        reasoning_all = load_dataset(f"{HF_USERNAME}/reasoning_all")["train"]
        featurized_questions = load_dataset(f"{HF_USERNAME}/train_featurized")["train"]
    else:
        reasoning_all = load_dataset("qfq/genminiall")["train"]
        featurized_questions = load_dataset("qfq/train_featurized")["train"]

    featurized_dict = {}
    for example in tqdm(featurized_questions):
        qhash = utils.question_hash(example["question"])
        example.pop("solution")
        example.pop("question")
        example.pop("cot_type")
        example.pop("source_type")
        example.pop("metadata")
        example.pop("cot")
        featurized_dict[qhash] = example

    reasoning_all = drop_na(reasoning_all, featurized_dict)

    logging.info(f"Length of reasoning_all: {len(reasoning_all)}")

    reasoning_all = drop_weird_string_patterns(reasoning_all)
    reasoning_all = drop_qwen_correct(reasoning_all, featurized_dict)

    reasoning_all_qhashes = set(
        [utils.question_hash(example["question"]) for example in reasoning_all]
    )
    featurized_dict = {
        k: v for k, v in featurized_dict.items() if k in reasoning_all_qhashes
    }
    print("Length of featurized_dict: ", len(featurized_dict))
    print("Length of reasoning_all: ", len(reasoning_all))

    if HF_USERNAME:
        reasoning_all.push_to_hub(f"{HF_USERNAME}/reasoning_all_hardfiltered_v2")
    else:
        reasoning_all.save_to_disk("results/reasoning_all_hardfiltered_v2")
