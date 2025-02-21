from transformers import AutoTokenizer
import utils
from tqdm import tqdm
import os
from dotenv import load_dotenv
from datasets import load_dataset
import argparse
from vllm import LLM, SamplingParams
import configuration
from huggingface_hub import hf_hub_download
import logging
from datetime import datetime
from datasets import Dataset
from glob import glob

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
logging.basicConfig(level=logging.INFO)


def _qwen_forward(
    prompts: list[str],
    model: LLM,
    sampling_params: SamplingParams,
):
    outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
    result = []
    for output in outputs:
        result.append(output.outputs[0].text)
    return result


def difficulty_classification(
    model_name: str, model: LLM, sampling_params: SamplingParams, batch_size: int
):
    if HF_USERNAME:
        questions = load_dataset(f"{HF_USERNAME}/s50k")["train"]["question"]
    else:
        questions = load_dataset("qfq/train")["train"]["question"]

    pretty_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = []
    result_dict = {}
    for i in tqdm(range(0, len(questions), batch_size)):
        question_batch = questions[i : i + batch_size]
        for question in question_batch:
            result_dict[utils.question_hash(question)] = None
            dialog = [{"role": "user", "content": question}]
            prompts.append(
                f"{tokenizer.apply_chat_template(dialog, tokenize=False)}<|im_start|>assistant\n"
            )
        batch = prompts[i : i + batch_size]
        results = _qwen_forward(batch, model, sampling_params)
        for question, result in zip(question_batch, results):
            result_dict[utils.question_hash(question)] = result

        logging.info(
            f"Processed batch {i // batch_size + 1}/{len(questions) // batch_size + 1}"
        )

    utils.jdump(result_dict, f"results/difficulty_classification/{pretty_name}.json")


def assemble_output(model_name: str, upload: bool = False):
    pretty_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    output = {}
    output.update(utils.jload(f"results/difficulty_classification/{pretty_name}.json"))
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/s50k")["train"]
    else:
        dataset = load_dataset("qfq/train")["train"]

    key_map_dataset = {}
    for example in tqdm(dataset, desc="Mapping dataset to hash"):
        key_map_dataset[utils.question_hash(example["question"])] = example
    result = []
    for qhash, attempt in tqdm(output.items(), desc="Creating output json"):
        if qhash in key_map_dataset:
            example = dict(
                question=key_map_dataset[qhash]["question"],
                solution=key_map_dataset[qhash]["solution"],
                attempt=attempt,
            )
            utils.jdump(
                example,
                f"results/difficulty_classification/{pretty_name}/grading_input/{qhash}.json",
            )
            result.append(example)
    if upload:
        new_dataset = []
        for example in dataset:
            example[pretty_name] = output[utils.question_hash(example["question"])]
            new_dataset.append(example)
        new_dataset = Dataset.from_list(new_dataset)
        if HF_USERNAME:
            new_dataset.push_to_hub(f"{HF_USERNAME}/train_{pretty_name}_inference")
        else:
            new_dataset.save_to_disk(f"results/train_{pretty_name}_inference")
    utils.jdump(
        result, f"results/difficulty_classification/{pretty_name}/inference_output.json"
    )


def assemble_reasoning_output(reasoning_model_name):
    pretty_name = reasoning_model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    jsons = [
        f
        for f in glob(f"results/reasoning/{reasoning_model_name}/*.json")
        if os.path.basename(f) != "args.json"
    ]
    if HF_USERNAME:
        dataset = load_dataset(f"{HF_USERNAME}/s50k")["train"]
    else:
        dataset = load_dataset("qfq/train")["train"]

    key_map_dataset = {}
    for example in tqdm(dataset, desc="Mapping dataset to hash"):
        key_map_dataset[utils.question_hash(example["question"])] = example
    for json_path in tqdm(jsons, desc="Creating grading input"):
        qdict = utils.jload(json_path)
        qhash = qdict["question_hash"]
        if qhash in key_map_dataset:
            new_qdict = dict(
                question=qdict["question"],
                solution=key_map_dataset[qhash]["solution"],
                attempt=qdict["response"],
            )
            utils.jdump(
                new_qdict,
                f"results/difficulty_classification/{pretty_name}/grading_input/{qhash}.json",
            )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Instruction model name in vLLM(default: Qwen/Qwen2.5-7B-Instruct)",
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
        default="auto",
        help="(default: auto)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="(default: 0.05)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="(default: 32768)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size for parallel inference(default: 2)",
    )
    parser.add_argument(
        "--reasoning_model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="reasoning model name in generate_reasoning.py(default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)",
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

    pretty_name = args.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    utils.jdump(
        vars(args), f"args/difficulty_classification/{pretty_name}-{current_time}.json"
    )

    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        seed=configuration.seed_number,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    difficulty_classification(args.model_name, model, sampling_params, args.batch_size)
    assemble_output(args.model_name)

    reasoning_output_dir = "results/difficulty_classification/reasoning"
    if os.path.exists(reasoning_output_dir) and os.path.isdir(reasoning_output_dir):
        pass
    else:
        assemble_reasoning_output(args.reasoning_model_name)
