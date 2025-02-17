import utils
import time
import logging
from datasets import load_dataset
import configuration
import constants
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import random
from dotenv import load_dotenv
import os

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
logging.basicConfig(level=logging.INFO)


def deepseek_qa(prompt: str, model: str):
    max_attempts = 1000
    answer = None
    attempts = 0
    while answer is None and attempts < max_attempts:
        try:
            answer = utils.ask_model(prompt, model)
            attempts += 1
            thinking, answer = answer["choices"][0]["message"]["content"].split(
                constants.DEEPSEEK_REASONING_TOKEN
            )
        except Exception as e:
            print(f"Exception: {str(e)}")
            time.sleep(60)

    return thinking.strip(), answer.strip()


def process_question(question: str, model: str, subdir: str):
    qhash = utils.question_hash(question)
    logging.info(f"Processing question {qhash}")
    thinking, response = deepseek_qa(question, model)
    result = dict(
        question_hash=qhash, question=question, thinking=thinking, response=response
    )
    utils.jdump(result, f"results/deepseek/{subdir}/{qhash}.json")
    logging.info(f"Processed question {qhash}")


def generate_deepseek(model):
    if HF_USERNAME:
        questions = load_dataset(f"{HF_USERNAME}/s50k")["train"]["question"]
    else:
        questions = load_dataset("qfq/train")["train"]["question"]

    random.seed(configuration.seed_number)
    random.shuffle(questions)
    logging.info(f"Processing {len(questions)} total questions")
    subdir = model
    existing_json = glob(f"results/deepseek/{subdir}/*.json")
    existing_qhash_list = [
        jsonpath.split("/")[-1].split(".")[0] for jsonpath in existing_json
    ]
    logging.info(f"Found {len(existing_qhash_list)} existing questions")
    questions = [
        question
        for question in questions
        if utils.question_hash(question) not in existing_qhash_list
    ]
    logging.info(
        f"{len(questions)} questions left after filtering existing question hashes"
    )
    process_map = partial(process_question, model=model, subdir=subdir)
    with ProcessPoolExecutor() as executor:
        executor.map(process_map, questions)


if __name__ == "__main__":
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    generate_deepseek(model)
