import processing

seed_number = 42

DS_TO_SELECTION = [
    {
        "ds_name": "MATH",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "OlympicArena",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "TheoremQA",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "NuminaMath",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "Omni-MATH",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "SciEval",
        "selection_strategy": processing.ScievalSelection(),
        "sampling_strategy": processing.RandomSampling(),
        "n_samples": 250,
    },
    {
        "ds_name": "OlympiadBench",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "JEEBench",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "AGIEval",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "StatsQual",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "GPQA",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "XWord",
        "selection_strategy": None,
        "sampling_strategy": processing.RandomSampling(),
        "n_samples": 1000,
    },
    {
        "ds_name": "USACO",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "Quant",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
    {
        "ds_name": "LiveCodeBench",
        "selection_strategy": None,
        "sampling_strategy": None,
        "n_samples": None,
    },
]
