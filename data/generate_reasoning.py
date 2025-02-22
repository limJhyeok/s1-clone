import utils
from datetime import datetime
import logging
from datasets import load_dataset, Dataset
import configuration
import constants
from glob import glob
import random
from dotenv import load_dotenv
import os
import argparse
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from tqdm import tqdm


load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
logging.basicConfig(level=logging.INFO)


def deepseek_qa_batch(
    prompts: list, model: LLM, sampling_params: SamplingParams = None
):
    max_attempts = 50
    results = [{"thinking": "", "answer": ""} for _ in range(len(prompts))]
    attempts = 0

    while (
        any(
            (result["thinking"] == "") or (result["answer"] == "") for result in results
        )
        and attempts < max_attempts
    ):
        try:
            remaining_indices = [
                i
                for i, result in enumerate(results)
                if (result["thinking"] == "") or (result["answer"] == "")
            ]
            remaining_prompts = [prompts[i] for i in remaining_indices]

            if not remaining_prompts:
                break

            response_batch = model.generate(remaining_prompts, sampling_params)

            for idx, response in zip(remaining_indices, response_batch):
                text = response.outputs[0].text
                if constants.DEEPSEEK_REASONING_TOKEN in text:
                    parts = text.split(constants.DEEPSEEK_REASONING_TOKEN, 1)
                    thinking = parts[0].strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                else:
                    thinking = ""
                    answer = text.strip()

                results[idx]["thinking"] = thinking
                results[idx]["answer"] = answer
        except Exception as e:
            logging.error(f"Batch processing error: {e}")

        attempts += 1
        if attempts >= max_attempts:
            logging.warning("Reached maximum attempts. Some results may be incomplete.")
            break

    return results


def process_questions_batch(
    questions: list[str],
    formatted_questions: list[str],
    model: LLM,
    subdir: str,
    sampling_params: SamplingParams,
):
    qhash_list = [utils.question_hash(q) for q in questions]
    logging.info(f"Processing {len(qhash_list)} questions")

    results = deepseek_qa_batch(formatted_questions, model, sampling_params)

    for qhash, question, (result) in zip(qhash_list, questions, results):
        row = dict(
            question_hash=qhash,
            question=question,
            thinking=result["thinking"],
            response=result["answer"],
        )
        utils.jdump(row, f"results/reasoning/{subdir}/{qhash}.json")
        logging.info(f"Processed question {qhash}")


def generate_deepseek_batch(
    model_name: str, model: LLM, batch_size: int, sampling_params: SamplingParams
):
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/s50k")["train"]
        questions = dataset["question"]
    else:
        dataset = load_dataset("qfq/train")["train"]
        questions = dataset["question"]

    random.seed(configuration.seed_number)
    random.shuffle(questions)
    logging.info(f"Processing {len(questions)} total questions")
    subdir = model_name

    existing_json = glob(f"results/reasoning/{subdir}/*.json")
    existing_qhash_list = {
        jsonpath.split("/")[-1].split(".")[0] for jsonpath in existing_json
    }

    questions = [
        q for q in questions if utils.question_hash(q) not in existing_qhash_list
    ]

    formatted_questions = [
        formatting_for_deepseek_r1(q, subject=d["cot_type"])
        for q, d in zip(dataset["question"], dataset)
        if utils.question_hash(q) not in existing_qhash_list
    ]

    for i in range(0, len(formatted_questions), batch_size):
        original_batch = questions[i : i + batch_size]
        formatted_batch = formatted_questions[i : i + batch_size]
        process_questions_batch(
            original_batch, formatted_batch, model, subdir, sampling_params
        )
        logging.info(
            f"Processed batch {i // batch_size + 1}/{len(formatted_questions) // batch_size + 1}"
        )


def formatting_for_deepseek_r1(prompt: str, subject: str = None) -> str:
    if subject == "math":
        return formatting_for_math_in_deepseek_r1(prompt)
    else:
        return formatting_for_general_in_deepseek_r1(prompt)


def formatting_for_general_in_deepseek_r1(prompt: str) -> str:
    return prompt + "<think>\n"


def formatting_for_math_in_deepseek_r1(prompt: str) -> str:
    return (
        prompt
        + "Please reason step by step, and put your final answer within \boxed{}."
    )


def upload_reasoning_result(model_name: str):
    jsons = glob(f"results/reasoning/{model_name}/*.json")
    if HF_USERNAME:
        all_train = load_dataset(f"{HF_USERNAME}/s50k")["train"]
    else:
        all_train = load_dataset("qfq/train")["train"]
    all_train_dict = {}
    for example in tqdm(all_train):
        all_train_dict[utils.question_hash(example["question"])] = example
    results = []
    for json_path in tqdm(jsons):
        qdict = utils.jload(json_path)
        qhash = qdict["question_hash"]
        if qhash in all_train_dict:
            all_train_example = all_train_dict[qhash]
            all_train_example["thinking_trajectories"] = [qdict["thinking"]]
            all_train_example["attempt"] = qdict["response"]
            results.append(all_train_example)
    dataset = Dataset.from_list(results)
    if HF_USERNAME:
        dataset.push_to_hub(f"{HF_USERNAME}/reasoning_all")
    else:
        dataset.save_to_disk("results/reasoning_all")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Reasoning model name in vLLM(default: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)",
    )
    parser.add_argument(
        "--max-model_len",
        type=int,
        default=16384,
        help="(default: 16384)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="number of GPU machine (default: 4)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Use bfloat16 for better performance on H100 GPUs (default: bfloat16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="(default: 0.6)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=15000,
        help="(default: 15,000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="batch size for parallel inference(default: 2048)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="(default: 0.95)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    try:
        config_path = hf_hub_download(args.model_name, "config.json")
        print(f"Config file found at: {config_path}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    utils.jdump(vars(args), f"args/reasoning/{args.model_name}-{current_time}.json")

    model = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        seed=configuration.seed_number,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens
    )

    generate_deepseek_batch(args.model_name, model, args.batch_size, sampling_params)
    upload_reasoning_result(args.model_name)
