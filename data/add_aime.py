import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets, Dataset

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")


def process_example(example):
    year = int(example["ID"].split("-")[0])
    if year <= 2022:
        solution = example.pop("Answer")
        question = example.pop("Question")
        cot_type = "math"
        source_type = "qq8933/AIME_1983_2024"
        metadata = str(example.copy())
        result = {
            "question": question,
            "solution": solution,
            "cot_type": cot_type,
            "source_type": source_type,
            "metadata": metadata,
        }
        return result
    else:
        return None


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_s1",
        type=bool,
        default=False,
        help="Whether to use s1K of official huggingface repo for downloading. if False: download from your huggingface(default: False)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    if args.use_s1:
        dataset = load_dataset("simplescaling/s1K")["train"]
    else:
        dataset = load_dataset(f"{HF_USERNAME}/s50k")["train"]

    aime = load_dataset("qq8933/AIME_1983_2024")["train"]
    aime_dataset = []
    for example in aime:
        result = process_example(example)
        if result is not None:
            aime_dataset.append(result)
    aime_dataset = Dataset.from_list(aime_dataset)
    new_dataset = concatenate_datasets([dataset, aime_dataset])
    new_dataset.push_to_hub(f"{HF_USERNAME}/s50k")
