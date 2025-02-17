import random
import datasets
from tqdm import tqdm
from dotenv import load_dotenv
import configuration
import processing
from functools import partial
import os
import utils

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")

if __name__ == "__main__":
    log_path = "./my_59k.log"
    random.seed(configuration.seed_number)

    test_datasets = {
        "AI-MO/aimo-validation-aime": {"split": "train", "question_field": "problem"},
        "Idavidrein/gpqa": {"split": "train", "question_field": "Question"},
        "qfq/openaimath": {"split": "test", "question_field": "problem"},
        "livecodebench/code_generation_lite": {
            "split": "test",
            "question_field": "question_content",
            "version_tag": "release_v4",
        },
    }

    test_questions = []
    for name, config in tqdm(test_datasets.items(), desc="Loading test questions"):
        test_questions.extend(processing.load_generic(name, **config)["question"])

    ds_all = []
    for kwargs in configuration.DS_TO_SELECTION:
        print(f"Processing: {kwargs.get('ds_name')}...")
        loader = processing.LoaderFactory.get_loader(**kwargs)
        ds = loader()

        # In the paper, authors collect 59K. To collect 59K I think this option should be turn off(commit id: 0b086e8fead36fc1c17319ea9c741be6f231a293)
        # ds = decontaminate_train_data(ds['question'], test_questions, ds, ngram_size=8)
        test_questions += ds["question"]
        ds_all.append(ds)

    ds = datasets.concatenate_datasets(ds_all)
    ds = ds.map(lambda x: {"cot": None, **x})
    memory = set()

    def is_unique(elem, column, memory):
        if elem[column] in memory:
            return False
        memory.add(elem[column])
        return True

    ds = ds.filter(partial(is_unique, column="question", memory=memory))
    ds.push_to_hub(f"{HF_USERNAME}/s50k")

    # log
    ds_dataframe = ds.to_pandas()
    utils.save_log_to_file(ds_dataframe, log_path)
