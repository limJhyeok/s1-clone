from tqdm import tqdm
import collections


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
