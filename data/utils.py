from tqdm import tqdm
import collections
import requests
import json
from hashlib import sha256
import os
import io


def normalize_string(text):
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def word_ngrams(text, n):
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def build_ngram_lookup(documents, ngram_size=13):
    """Build ngram lookup for documents."""
    print(f"Building {ngram_size}-gram lookup...")
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)

    return lookup


def find_contaminated_questions(test_lookup, train_lookup):
    """Find overlapping documents based on ngram matches."""
    contaminated_ids = set()
    matched_ngrams = []  # For debugging

    for ngram, test_doc_ids in tqdm(test_lookup.items(), desc="Checking overlaps"):
        if ngram in train_lookup:
            contaminated_ids.update(test_doc_ids)
            matched_ngrams.append(ngram)

    # Print some example matches for inspection
    if matched_ngrams:
        print("\nExample matching n-grams:")
        for ngram in matched_ngrams[:5]:  # Show first 5 matches
            print(f"  - {ngram}")

    return contaminated_ids


def decontaminate_train_data(train_questions, test_questions, ds, ngram_size=8):
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    test_lookup = build_ngram_lookup(test_questions, ngram_size)

    # Find contaminated questions
    contaminated_ids = find_contaminated_questions(train_lookup, test_lookup)

    # Remove contaminated examples
    not_contaminated_ids = set(range(len(train_questions))) - contaminated_ids
    ds = ds.select(list(not_contaminated_ids))
    print("\nDecontamination Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Contaminated questions: {len(contaminated_ids)}")
    print(
        f"Contamination rate: {(len(contaminated_ids) / len(train_questions) * 100):.2f}%"
    )
    print(f"Clean examples remaining: {len(ds)}")
    return ds


def save_log_to_file(ds_dataframe, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Total data size: {len(ds_dataframe)}\n\n")

        # 각 source_type별 개수
        source_type_counts = ds_dataframe["source_type"].value_counts()
        f.write("each source_type size:\n")
        f.write(source_type_counts.to_string() + "\n\n")

        # 문제 길이 분석
        ds_dataframe["question_length"] = ds_dataframe["question"].apply(len)
        f.write(
            f"average of question_length: {ds_dataframe['question_length'].mean():.2f}\n"
        )
        f.write(
            f"median of question_length: {ds_dataframe['question_length'].median():.2f}\n"
        )
        f.write(
            f"minimum of question_length: {ds_dataframe['question_length'].min()}\n"
        )
        f.write(
            f"maximum of question_length: {ds_dataframe['question_length'].max()}\n\n"
        )

    print(f"✅ log is saved: {file_path}")


def ask_model(question: str, model: str, port: int = 8000):
    """
    Sends a request to the model server and fetches a response.
    from https://medium.com/@hakimnaufal/trying-out-vllm-deepseek-r1-in-google-colab-a-quick-guide-a4fe682b8665
    """
    url = f"http://localhost:{port}/v1/chat/completions"  # Adjust the URL if different
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": question}]}

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.json()


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="utf-8")
    return f


def jdump(obj, f: str, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="utf-8")
    return f


def question_hash(question: str) -> str:
    return sha256(question.encode()).hexdigest()[:16]
