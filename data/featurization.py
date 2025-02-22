from datasets import load_dataset, Dataset
import utils
import inference_utils
import os
from glob import glob
import logging
from time import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
logging.basicConfig(level=logging.INFO)


def _process_question(question: str, msc_prompt: str, system_prompt: str):
    qhash = utils.question_hash(question)
    user_prompt = f"## Question\n{question}\n\n## Classification rubrics\n{msc_prompt}"
    result = None
    logging.info(f"Processing {qhash}")
    while result is None:
        completion, _ = inference_utils.apiqa(
            user_prompt, "claude-3-5-sonnet-20241022", system_prompt, json_format=False
        )

        if completion[-2:].isdigit() and len(completion[-2:]) == 2:
            result = completion[-2:]
        else:
            logging.info(f"Invalid response: {completion}")
            time.sleep(60)
    result_dict = dict(
        qhash=qhash, question=question, raw_response=completion, domain=result
    )
    utils.jdump(result_dict, f"results/domain_classification/{qhash}.json")
    logging.info(f"Processed {qhash}")


def do_domain_classification():
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/s50k")["train"]
    else:
        dataset = load_dataset("qfq/train")["train"]

    msc = utils.jload("data/msc.json")
    msc_prompt = "\n\n".join([node["prompt"] for node in msc])
    system_prompt = (
        "You are a helpful assistant that classifies questions into different subjects based on the provided classification rubrics. "
        "You will be given a question and a list of subjects. "
        "You need to classify the question into one of the subjects. "
        "If the question has multiple subjects, you should classify the question into the most relevant subject. "
        "Explain your reasoning, and end your response on a new line with two-digit code of the subject that the question belongs to."
    )
    questions = dataset["question"]
    existing_qhash = glob("results/domain_classification/*.json")
    existing_qhash = [os.path.basename(path).split(".")[0] for path in existing_qhash]
    logging.info(f"Found {len(existing_qhash)} existing questions")
    questions = [
        question
        for question in questions
        if utils.question_hash(question) not in existing_qhash
    ]
    logging.info(f"Processing {len(questions)} new questions")
    _process_question_map = partial(
        _process_question, msc_prompt=msc_prompt, system_prompt=system_prompt
    )
    with ProcessPoolExecutor(max_workers=100) as executor:
        executor.map(_process_question_map, questions)


def _process_example_grading(qhash: str, grading_prompt: str, response_dir: str):
    logging.info(f"Grading example {qhash}")
    # formating user prompt
    example = utils.jload(
        f"results/difficulty_classification/{response_dir}/grading_input/{qhash}.json"
    )
    if example["solution"] is None:
        result = {
            "question": example["question"],
            "solution": example["solution"],
            "attempt": example["attempt"],
            "grade": None,
            "reason": None,
        }
    else:
        user_prompt = f"## Problem\n{example['question']}\n\n## Attempt\n{example['attempt']}\n\n## Correct answer\n{example['solution']}"
        # do grading
        grade = None
        while grade is None:
            completion, _ = inference_utils.apiqa(
                user_prompt,
                "claude-3-5-sonnet-20241022",
                grading_prompt,
                json_format=False,
            )
            if completion.split("\n")[-1] in ["Yes", "No"]:
                grade = completion.split("\n")[-1]
            else:
                logging.info(f"Invalid response: {completion}")
                time.sleep(60)
        # save result
        result = {
            "question": example["question"],
            "solution": example["solution"],
            "attempt": example["attempt"],
            "grade": grade,
            "reason": completion,
        }
    utils.jdump(
        result,
        f"results/difficulty_classification/{response_dir}/grading_output/{qhash}.json",
    )
    logging.info(f"Finished grading example {qhash}")


def do_grading(model_name: str = "Qwen/Qwen2.5-32B-Instruct"):
    pretty_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    if HF_USERNAME:
        true_qhashes = [
            utils.question_hash(q)
            for q in load_dataset(f"{HF_USERNAME}/train")["train"]["question"]
        ]
    else:
        true_qhashes = [
            utils.question_hash(q)
            for q in load_dataset("qfq/train")["train"]["question"]
        ]
    jsons = glob(
        f"results/difficulty_classification/{pretty_name}/grading_input/*.json"
    )
    qhashes = [os.path.basename(json).split(".")[0] for json in jsons]
    qhashes = [qhash for qhash in qhashes if qhash in true_qhashes]
    grading_prompt = utils.tload("data/grading.txt")
    logging.info(f"Grading {len(qhashes)} examples")
    graded_qhashes = glob(
        f"results/difficulty_classification/{pretty_name}/grading_output/*.json"
    )
    graded_qhashes = [os.path.basename(json).split(".")[0] for json in graded_qhashes]
    qhashes = [qhash for qhash in qhashes if qhash not in graded_qhashes]
    logging.info(f"Grading {len(qhashes)} new examples")
    _process_example_grading_partial = partial(
        _process_example_grading,
        grading_prompt=grading_prompt,
        response_dir=pretty_name,
    )
    with ProcessPoolExecutor() as executor:
        executor.map(_process_example_grading_partial, qhashes)


def upload_grading(model_name: str = "Qwen/Qwen2.5-32B-Instruct"):
    pretty_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")

    if pretty_name == "Qwen_Qwen2_5_32B_Instruct":
        col_name = "isqwen32bcorrect"
    elif pretty_name == "Qwen_Qwen2_5_7B_Instruct":
        col_name = "isqwen7bcorrect"
    elif pretty_name == "Qwen_QwQ_32B_Preview":
        col_name = "isqwqcorrect"
    elif pretty_name == "genmini":
        col_name = "isgenminicorrect"
    elif pretty_name == "deepseek_ai_DeepSeek_R1_Distill_Qwen_1_5B":
        col_name = "isdeepseek1_5bcorrect"
    elif pretty_name == "deepseek_ai_DeepSeek_R1_Distill_Qwen_7B":
        col_name = "isdeepseek7bcorrect"
    elif pretty_name == "deepseek_ai_DeepSeek_R1_Distill_Qwen_32B":
        col_name = "isdeepseek32bcorrect"
    else:
        raise ValueError(f"Invalid response directory: {pretty_name}")
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/train_featurized")["train"]
    else:
        dataset = load_dataset("qfq/train_featurized")["train"]

    def _add_grade(example):
        qhash = utils.question_hash(example["question"])
        grade_path = f"results/difficulty_classification/{pretty_name}/grading_output/{qhash}.json"
        if os.path.exists(grade_path):
            claude_grade = utils.jload(grade_path)["grade"]
            if claude_grade in ["Yes", "No"]:
                example[col_name] = claude_grade == "Yes"
            else:
                assert example["solution"] is None
                example[col_name] = None
        else:
            example[col_name] = None
        return example

    result = []
    for example in tqdm(dataset):
        try:
            result.append(_add_grade(example))
        except Exception as e:
            logging.info(
                f"Error processing {utils.question_hash(example['question'])}: {e}"
            )
            import pdb

            pdb.set_trace()
    dataset = Dataset.from_list(result)
    if HF_USERNAME:
        dataset.push_to_hub(f"{HF_USERNAME}/train_featurized")
    else:
        dataset.save_to_disk("results/train_featurized")


def upload_domain():
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/s50k")["train"]
    else:
        dataset = load_dataset("qfq/train")["train"]

    msc = utils.jload("data/msc.json")
    msc_dict = {}
    for subject in msc:
        msc_dict[subject["code"]] = subject["title"]

    def _add_domain(example):
        try:
            qhash = utils.question_hash(example["question"])
            domain_path = f"results/domain_classification/{qhash}.json"
            example["domain"] = msc_dict[utils.jload(domain_path)["domain"]]
        except Exception as e:
            logging.info(
                f"Error processing {utils.question_hash(example['question'])}: {e}"
            )
            example["domain"] = None
        return example

    result = []
    for example in tqdm(dataset):
        result.append(_add_domain(example))
    dataset = Dataset.from_list(result)
    if HF_USERNAME:
        dataset.push_to_hub(f"{HF_USERNAME}/train_featurized")
    else:
        dataset.save_to_disk("results/train_featurized")


def upload_length():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/reasoning_all")["train"]
    else:
        dataset = load_dataset("qfq/genminiall")["train"]

    def _compute_length(example):
        tokens = tokenizer.encode(example["thinking_trajectories"][0])
        return dict(len=len(tokens))

    dataset = dataset.map(_compute_length)
    genmini_length_dict = {}
    for example in tqdm(dataset):
        genmini_length_dict[utils.question_hash(example["question"])] = example["len"]
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/train_featurized")["train"]
    else:
        dataset = load_dataset("qfq/train_featurized")["train"]
    result = []
    for example in tqdm(dataset):
        qhash = utils.question_hash(example["question"])
        length = None
        if qhash in genmini_length_dict:
            length = genmini_length_dict[qhash]
        example["genmini_length"] = length
        result.append(example)
    dataset = Dataset.from_list(result)
    if HF_USERNAME:
        dataset.push_to_hub(f"{HF_USERNAME}/train_featurized")
    else:
        dataset.save_to_disk("results/train_featurized")


if __name__ == "__main__":
    do_domain_classification()
    upload_domain()

    do_grading("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    upload_grading("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    do_grading("Qwen/Qwen2.5-7B-Instruct")
    upload_grading("Qwen/Qwen2.5-7B-Instruct")

    do_grading("Qwen/Qwen2.5-32B-Instruct")
    upload_grading("Qwen/Qwen2.5-32B-Instruct")

    upload_length()
