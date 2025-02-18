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
import argparse
from vllm import LLM, SamplingParams


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


def deepseek_qa_batch(
    prompts: list, model: LLM, sampling_params: SamplingParams = None
):
    max_attempts = 3
    batch_size = len(prompts)
    answers = [None] * batch_size
    attempts = 0

    while any(answer is None for answer in answers) and attempts < max_attempts:
        try:
            response_batch = model.generate(prompts, sampling_params)
            for i, response in enumerate(response_batch):
                text = response.outputs[0].text
                if constants.DEEPSEEK_REASONING_TOKEN in text:
                    thinking, answer = text.split(constants.DEEPSEEK_REASONING_TOKEN)
                    answers[i] = (thinking.strip(), answer.strip())
                else:
                    answers[i] = ("", text.strip())
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
        attempts += 1

    return answers


def process_question(question: str, model: str, subdir: str):
    qhash = utils.question_hash(question)
    logging.info(f"Processing question {qhash}")
    thinking, response = deepseek_qa(question, model)
    result = dict(
        question_hash=qhash, question=question, thinking=thinking, response=response
    )
    utils.jdump(result, f"results/deepseek/{subdir}/{qhash}.json")
    logging.info(f"Processed question {qhash}")


def process_questions_batch(
    questions: list[str], model: LLM, subdir: str, sampling_params: SamplingParams
):
    qhash_list = [utils.question_hash(q) for q in questions]
    logging.info(f"Processing {len(qhash_list)} questions")

    results = deepseek_qa_batch(questions, model, sampling_params)

    for qhash, question, (thinking, response) in zip(qhash_list, questions, results):
        result = dict(
            question_hash=qhash,
            question=question,
            thinking=thinking,
            response=response,
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


def generate_deepseek_batch(
    model_name: str, model: LLM, batch_size: int, sampling_params: SamplingParams
):
    if HF_USERNAME:
        questions = load_dataset(f"{HF_USERNAME}/s50k")["train"]["question"]
    else:
        questions = load_dataset("qfq/train")["train"]["question"]

    random.seed(configuration.SEED_NUMBER)
    random.shuffle(questions)
    logging.info(f"Processing {len(questions)} total questions")
    subdir = model_name

    existing_json = glob(f"results/deepseek/{subdir}/*.json")
    existing_qhash_list = {
        jsonpath.split("/")[-1].split(".")[0] for jsonpath in existing_json
    }
    questions = [
        q for q in questions if utils.question_hash(q) not in existing_qhash_list
    ]
    logging.info(f"{len(questions)} questions left after filtering")

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        process_questions_batch(batch, model_name, subdir)
        logging.info(
            f"Processed batch {i // batch_size + 1}/{len(questions) // batch_size + 1}"
        )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Reasoning model name in vLLM(default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="(default: 1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="(default: float16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="(default: 0.6)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size for parallel inference(default: 2)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    utils.jdump(vars(args), f"results/deepseek/{args.model_name}/args.json")

    model = LLM(
        args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
    )

    generate_deepseek_batch(args.model_name, model, args.batch_size, sampling_params)
