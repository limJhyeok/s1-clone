from tqdm import tqdm
from dotenv import load_dotenv
import os
from datasets import load_dataset, Dataset
import utils
import logging
import re
from collections import Counter
import numpy as np
from tokenization import mathcot_sft

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

    selected_qhashes = set()
    for example in reasoning_all:
        qhash = utils.question_hash(example["question"])
        if featurized_dict[qhash]["isgenminicorrect"]:
            if example["source_type"] in ["Idavidrein/gpqa", "qq8933/AIME_1983_2024"]:
                selected_qhashes.add(qhash)
            elif "qfq/openaimath" in example["source_type"]:
                if featurized_dict[qhash]["genmini_length"] > 5600:
                    selected_qhashes.add(qhash)

    # Diversity Sampling
    gpqa_domain = [
        featurized_dict[utils.question_hash(x["question"])]["domain"]
        for x in reasoning_all.filter(lambda x: x["source_type"] == "Idavidrein/gpqa")
    ]
    gpqa_domain = Counter(gpqa_domain)
    aime_domain = [
        featurized_dict[utils.question_hash(x["question"])]["domain"]
        for x in reasoning_all.filter(
            lambda x: x["source_type"] == "qq8933/AIME_1983_2024"
        )
    ]
    aime_domain = Counter(aime_domain)
    enlarge_ratio = 0.5 * sum(aime_domain.values()) / sum(gpqa_domain.values())
    for domain in gpqa_domain:
        gpqa_domain[domain] = int(gpqa_domain[domain] * enlarge_ratio)
    benchmark_domain = gpqa_domain + aime_domain
    benchmark_domains, benchmark_weights = (
        list(benchmark_domain.keys()),
        list(benchmark_domain.values()),
    )
    benchmark_weights = np.array(benchmark_weights) / sum(benchmark_weights)

    def benchmark_sample(benchmark_domains, benchmark_weights):
        return np.random.choice(benchmark_domains, size=1, p=benchmark_weights)[0]

    all_domains = list(
        set(
            [
                featurized_dict[utils.question_hash(example["question"])]["domain"]
                for example in reasoning_all
            ]
        )
    )

    def uniform_sample(all_domains):
        return np.random.choice(all_domains, size=1)[0]

    questions_ordered_by_domain = {}
    for domain in tqdm(all_domains):
        questions_ordered_by_domain[domain] = {
            k: v for k, v in featurized_dict.items() if v["domain"] == domain
        }

    # Powerlaw length sampling
    pbar = tqdm(initial=len(selected_qhashes), total=1000, desc="Sampling questions")
    while len(selected_qhashes) < 1000:
        # first sample 300 uniformly over all domains
        if len(selected_qhashes) < 700:
            random_domain = uniform_sample(all_domains)
        else:
            random_domain = benchmark_sample(benchmark_domains, benchmark_weights)
        # Sort by chain length and take the longest one
        domain_examples = questions_ordered_by_domain[random_domain]
        qhashes = list(domain_examples.keys())
        lengths = np.array(
            [int(domain_examples[qhash]["genmini_length"]) for qhash in qhashes]
        )
        ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))
        length_weights = np.power(2.0, -ranks)
        length_weights = length_weights / length_weights.sum()
        selected_qhash = np.random.choice(qhashes, p=length_weights)
        selected_qhashes.add(selected_qhash)
        questions_ordered_by_domain[random_domain].pop(selected_qhash)
        if len(questions_ordered_by_domain[random_domain]) == 0:
            if random_domain in all_domains:
                all_domains.remove(random_domain)
            if random_domain in benchmark_domains:
                benchmark_weights = np.delete(
                    benchmark_weights, benchmark_domains.index(random_domain)
                )
                benchmark_weights = benchmark_weights / benchmark_weights.sum()
                benchmark_domains.remove(random_domain)
        pbar.update(1)
    pbar.close()

    print("Verify length of selected_qhashes:", len(selected_qhashes))
    sampled_reasoning_all = reasoning_all.filter(
        lambda x: utils.question_hash(x["question"]) in selected_qhashes
    )
    print("Verify length of sampled_reasoning_all:", len(sampled_reasoning_all))

    if HF_USERNAME:
        sampled_reasoning_all.push_to_hub(f"{HF_USERNAME}/s1K")
    else:
        sampled_reasoning_all.save_to_disk("results/s1k")

    if HF_USERNAME:
        download_data_path = f"{HF_USERNAME}/s1K"
        upload_data_path = f"{HF_USERNAME}/s1K_tokenized"
    else:
        download_data_path = "simplescaling/s1K"
        upload_data_path = None

    mathcot_sft(
        download_data_path=download_data_path,
        upload_data_path=upload_data_path,
        num_proc=80,
    )
