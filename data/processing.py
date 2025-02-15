from abc import ABC, abstractmethod
from collections import Counter
import constants
from dotenv import load_dotenv
import random
import datasets
from tqdm import tqdm

load_dotenv()
random.seed(42)


def load_generic(
    name,
    split,
    question_field="question",
    solution_field="solution",
    cot_type="math",
    version_tag=None,
):
    conf = "gpqa_diamond" if name == "Idavidrein/gpqa" else None
    ds = datasets.load_dataset(
        name, conf, version_tag=version_tag, trust_remote_code=True
    )[split]
    # Make metadata a string that can be loaded via literal_eval to avoid TypeError: Couldn't cast array of type list<item: string> to null
    ds = ds.map(
        lambda x: {
            "question": x.pop(question_field),
            "solution": x.pop(solution_field, None),
            "cot_type": cot_type,
            "source_type": name,
            "metadata": str(x),
        }
    )
    ds = ds.remove_columns(
        [c for c in ds.column_names if c not in constants.DS_COLUMNS]
    )
    return ds


class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, ds, n_examples):
        pass


class ScievalSelection(SelectionStrategy):
    def select(self, ds, n_examples):
        ### TMP ###
        samples_to_keep = ds.filter(
            lambda x: x["question"] in constants.TMP_QUESTIONS_TO_KEEP
        )
        n_examples -= len(samples_to_keep)

        import math

        tasks = set([eval(x)["task_name"] for x in ds["metadata"]])
        tasks_to_num_samples = {
            task: math.ceil(n_examples / len(tasks)) for task in tasks
        }
        # Only SocraticQA has topics
        topics = set([eval(x)["topic"] for x in ds["metadata"]])
        socratic_qa_samples = tasks_to_num_samples["SocraticQA"]
        topics_to_num_samples = {
            topic: math.ceil(socratic_qa_samples / len(topics)) for topic in topics
        }
        selected_examples = []
        ds = ds.shuffle(seed=42)
        for i, ex in enumerate(ds):
            meta = eval(ex["metadata"])
            if meta["topic"] in topics:
                if topics_to_num_samples[meta["topic"]] > 0:
                    selected_examples.append(ex)
                    topics_to_num_samples[meta["topic"]] -= 1
                    tasks_to_num_samples["SocraticQA"] -= 1
            if (meta["task_name"] in tasks) and (
                tasks_to_num_samples[meta["task_name"]] > 0
            ):
                selected_examples.append(ex)
                tasks_to_num_samples[meta["task_name"]] -= 1
            if len(selected_examples) == n_examples:
                break

        ### TMP ###
        ds = datasets.concatenate_datasets(
            [datasets.Dataset.from_list(selected_examples), samples_to_keep]
        )
        return ds


class OmniMATHSelection(SelectionStrategy):
    def cleansing(self, ds):
        return list(ds)

    def select(self, ds, n_examples):
        clean_examples = self.cleansing(ds)
        ### TODO: Filter out BAD_OMNIMATH_SAMPLES
        random.shuffle(clean_examples)
        all_subdomains = list()
        all_difficulties = list()
        # get all subdomains and difficulties
        for ex in tqdm(clean_examples, desc="Selecting examples"):
            meta = eval(ex["metadata"])
            try:
                domain = meta["domain"][0].split(" -> ")
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error: {e}, not domain found")
                clean_examples.remove(ex)
                continue
            try:
                subdomain = domain[2]
            except IndexError as e:
                print(f"Error: {e}, not subdomain found")
                clean_examples.remove(ex)
                continue
            if len(subdomain) > 30:
                continue
            difficulty = meta["difficulty"]
            all_subdomains.append(subdomain)
            all_difficulties.append(difficulty)

        # make a counter for each subdomain and difficulty
        subdomain_counter = Counter(all_subdomains)
        # difficulty_counter = Counter(all_difficulties)

        # sort in descending order of subdomain count
        subdomain_counter = sorted(
            subdomain_counter.items(), key=lambda x: x[1], reverse=False
        )
        print("subdomain_counter: ", len(subdomain_counter))
        # select n_examples from clean_examples covering diverse domains (select examples from each subdomain) and hard questions (difficulty > 6)
        selected_examples = []
        selected_subdomains = []
        selected_difficulties = []
        for subdomain, _ in subdomain_counter:
            count = 0
            for ex in clean_examples:
                meta = eval(ex["metadata"])
                if (
                    meta["domain"][0].split(" -> ")[2] == subdomain
                    and meta["difficulty"] > 6.5
                ):
                    selected_examples.append(ex)
                    selected_subdomains.append(subdomain)
                    selected_difficulties.append(meta["difficulty"])
                    count += 1
                    if len(selected_examples) == n_examples:
                        return (
                            selected_examples,
                            selected_subdomains,
                            selected_difficulties,
                        )
                    # no more than 50 examples per subdomain
                    # if count > 40: break
        return selected_examples, selected_subdomains, selected_difficulties


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, ds, n_samples):
        pass


class RandomSampling(SamplingStrategy):
    def sample(self, ds, n_samples):
        questions_to_keep = ds.filter(
            lambda x: x["question"] in constants.TMP_QUESTIONS_TO_KEEP
        )
        n_samples -= len(questions_to_keep)
        ds = ds.select(range(n_samples))
        ds = datasets.concatenate_datasets([ds, questions_to_keep])
        return ds


class HFLoad(ABC):
    def __init__(
        self,
        selection_strategy: SelectionStrategy = None,
        sampling_strategy: SamplingStrategy = None,
        n_samples: int = None,
    ):
        self.selection_strategy = selection_strategy
        self.sampling_strategy = sampling_strategy
        self.n_samples = n_samples

    @abstractmethod
    def load_fn(self):
        pass

    def apply_selection(self, ds, n_examples: int):
        if self.selection_strategy:
            return self.selection_strategy.select(ds, n_examples)
        return ds

    def apply_sampling(self, ds, n_examples: int):
        if self.sampling_strategy:
            return self.sampling_strategy.sample(ds, n_examples)
        return ds

    def __call__(self):
        ds = self.load_fn()
        if self.selection_strategy and self.n_samples:
            ds = self.apply_selection(ds, self.n_samples)
        else:
            ds = ds.shuffle(seed=42)
            if self.n_samples:
                ds = self.apply_sampling(ds, self.n_samples)
        return ds


class OpenAIMathLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("simplescaling/openaimath", trust_remote_code=True)[
            "train"
        ]
        ds = ds.map(
            lambda x: {
                "question": x.pop("problem"),
                "solution": x.pop("solution"),
                "cot_type": "math",
                "source_type": "simplescaling/openaimath/" + x["subject"],
                "metadata": str(x),
            },
            writer_batch_size=constants.LARGE_DATASET_WRITER_BATCH_SIZE,
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class OlympicArenaLoader(HFLoad):
    def load_fn(self):
        confs = [
            "Math",
            "Physics",
            "Chemistry",
            "Biology",
            "Geography",
            "Astronomy",
            "CS",
        ]
        subject_to_o1domain = {"Math": "math", "CS": "coding"}
        ds = [
            datasets.load_dataset("GAIR/OlympicArena", c, trust_remote_code=True)
            for c in confs
        ]
        ds = datasets.concatenate_datasets(
            [d["test"] for d in ds] + [d["val"] for d in ds]
        )
        # Filter for EN & text-only
        ds = ds.filter(
            lambda x: (x["language"] == "EN") and (x["modality"] == "text-only")
        )
        ds = ds.map(
            lambda x: {
                "question": x.pop("problem"),
                "solution": x.pop("solution"),
                "cot_type": subject_to_o1domain.get(x["subject"], "science"),
                "source_type": "GAIR/OlympicArena/" + x["subject"],
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        ds = ds.shuffle(seed=42)
        return ds


class TheoremQALoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("TIGER-Lab/TheoremQA", trust_remote_code=True)[
            "test"
        ]
        ds = ds.filter(lambda x: x["Picture"] is None)
        ds = ds.map(
            lambda x: {
                "question": x.pop("Question"),
                "solution": x.pop("Answer"),
                "cot_type": "math",
                "source_type": "TIGER-Lab/TheoremQA/" + x["Answer_type"],
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        ds.shuffle(seed=42)
        return ds


class NuminaMathLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("AI-MO/NuminaMath-CoT", trust_remote_code=True)[
            "train"
        ]
        ds_aops = ds.filter(lambda x: x["source"] == "aops_forum")
        ds = ds.filter(lambda x: x["source"] != "aops_forum")

        ### TMP ###
        questions = datasets.load_dataset("qfq/numinamath_500", trust_remote_code=True)[
            "train"
        ]["problem"]
        ds = ds.filter(
            lambda x: (x["problem"] in questions)
            or (x["problem"] in constants.TMP_QUESTIONS_TO_KEEP)
        )
        ds = datasets.concatenate_datasets([ds, ds_aops])

        ds = ds.map(
            lambda x: {
                "question": x.pop("problem"),
                "solution": x.pop("solution"),
                "cot_type": "math",
                "source_type": "AI-MO/NuminaMath-CoT/" + x["source"],
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class OmniMATHLoader(HFLoad):
    def load_fn(self):
        ds = load_generic(
            name="KbsdJames/Omni-MATH", question_field="problem", split="test"
        )
        return ds


class SciEvalLoader(HFLoad):
    def load_fn(self):
        """
        Category Physics; Task: SocraticQA
        What is the moment of inertia of a pendulum with a mass of $2 kg$ that is $7  m$ from the pivot?\n\nA. 56 kgm^2\nB. 196 kgm^2\nC. 84 kgm^2\nD. 98 kgm^2\n\nAnswer:

        Category Chemistry; Task: SocraticQA
        What is the molecular geometry of the $PF_3$ molecule?\n\nA. Trigonal planar\nB. Bent\nC. Trigonal pyramidal\nD. Tetrahedral\n\nAnswer:
        Category Chemistry; Task: reagent selection
        Given the rest of reaction components:\nreactant 1: Ic1ccc2ncccc2c1\nreactant 2: Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1\nligand: c1ccc(P(c2ccccc2)c2ccccc2)cc1\nbase: C(=O)(O)[O-].[Na+]  \nSolvent list for selection:\nC1CCOC1,CN(C)C=O,CO\nOptimal solvent:

        Category Biology; Task: MedQA
        A 74-year-old man was admitted to the intensive care ward due to progressive dyspnea, cough with pink sputum, and diaphoresis. He had 2 myocardial infarctions at the age of 66 and 69 years and suffers from chronic heart failure. At the time of presentation, his vital signs are as follows: blood pressure 90/50 mm Hg, heart rate 108/min, respiratory rate 29/min, and temperature 35.5°C (95.9°F). On physical examination, the patient sits upright. He is lethargic and cyanotic. Lung auscultation reveals widespread bilateral fine rales. Cardiac examination is significant for S3, accentuation of the pulmonic component of S2, and a systolic murmur heard best at the apex of the heart. Soon after hospitalization, the patient develops ventricular fibrillation and dies despite adequate resuscitation measures. Which microscopic finding would you expect to see in this patient on autopsy?\n\nA. Brownish inclusions in the pulmonary macrophages on H&E staining\nB. Positive Prussian-blue staining of the kidney tissue\nC. Ground-glass hepatocytes\nD. Positive Congo-red staining of the cardiac tissue\n\nAnswer:
        Category Biology; Task: PubMedQA
        Polymorphisms in the oestrogen receptor 1 (ESR1) and oestrogen receptor 2 (ESR2) genes are associated with intermediate or endpoint markers of cardiovascular disease and with the efficacy of postmenopausal hormone therapy (HT). Contradictory findings have been described in the past and the role of these genetics variants remains unclear.\nA cross-sectional study was carried out with 266 postmenopausal women, of whom 115 received oral HT (HT+) and 151 did not receive any HT (HT-). We analysed three single-nucleotide polymorphisms (SNPs) in ESR1 (rs1801132, rs7757956 and rs2813544) and two in ESR2 (rs3020450 and rs7154455) and derived haplotypes with three additional polymorphisms that had been previously investigated by our group (ESR1 rs2234693 and ESR2 rs1256049 and rs4986938).\nThe ESR1 rs2813544 polymorphism was associated with low-density lipoprotein cholesterol (LDL-C) in HT+ postmenopausal women (p\u2009=\u20090.044; pC\u2009=\u20090.388), while one ESR2 gene haplotype was associated with total cholesterol (T-chol) (p\u2009=\u20090.015; pC\u2009=\u20090.090) and LDL-C in HT+ postmenopausal women (p\u2009=\u20090.021; pC\u2009=\u20090.126).\n\nAre polymorphisms in oestrogen receptors genes associated with lipid levels in response to hormone therapy?\n\nAnswer:
        Category Biology; Task: SocraticQA
        What substance is transported across the inner membrane of the mitochondria?\n\nA. Glucose\nB. Protons\nC. Oxygen\nD. Electrons\n\nAnswer:
        """
        ds = datasets.load_dataset("OpenDFM/SciEval", trust_remote_code=True)
        ds = datasets.concatenate_datasets([ds["test"], ds["validation"]])
        # As there's enough samples, filter out ones without answer (13011 out of 27533)
        ds = ds.filter(lambda x: x["answer"] is not None)

        # Remove the "\n\nAnswer:"; Replace "\nOptimal solvent:" with "\nWhat is the optimal solvent?"; "\nOptimal ligand:" with "\nWhat is the optimal ligand?"; "\nOptimal reactant" with "\nWhat is the optimal reactant?"
        def clean_question(x):
            x["question"] = (
                x["question"]
                .split("\n\nAnswer:")[0]
                .replace("\nOptimal solvent:", "\nWhat is the optimal solvent?")
                .replace("\nOptimal ligand:", "\nWhat is the optimal ligand?")
                .replace("\nOptimal reactant:", "\nWhat is the optimal reactant?")
            )
            return x

        ds = ds.map(clean_question)
        task_to_o1domain = {
            "SocraticQA": "science",
            "reagent selection": "science",
            "MedQA": "health science",
            "PubMedQA": "health science",
        }
        ds = ds.map(
            lambda x: {
                "question": x.pop("question"),
                "solution": x.pop("answer")[0],
                "cot_type": task_to_o1domain[x["task_name"]],
                "source_type": "OpenDFM/SciEval/"
                + x["category"]
                + "/"
                + x["type"]
                + "/"
                + x["task_name"],
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class OlympiadBenchLoader(HFLoad):
    def load_fn(self):
        # Only EN & TO (text-only); Both OE (open-ended) and TP (Theorem proof)
        confs = [
            "OE_TO_maths_en_COMP",
            "OE_TO_physics_en_COMP",
            "TP_TO_maths_en_COMP",
            "TP_TO_physics_en_COMP",
        ]
        # Multimodal: "OE_MM_maths_en_COMP", "OE_MM_physics_en_COMP", "TP_MM_maths_en_COMP", "TP_MM_physics_en_COMP"
        ds = [
            datasets.load_dataset("Hothan/OlympiadBench", c, trust_remote_code=True)[
                "train"
            ]
            for c in confs
        ]
        ds = datasets.concatenate_datasets(ds)
        ### TODO: Is solution ever longer than 1?
        ### TODO: forgot to add context to question
        # The physics one is also rather math-heavy
        ds = ds.map(
            lambda x: {
                "question": x.pop("question"),
                "solution": x.pop("solution")[0],
                "cot_type": "math",
                "source_type": "Hothan/OlympiadBench/"
                + x["question_type"]
                + "/"
                + x["subject"],
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class JEEBenchLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("daman1209arora/jeebench", trust_remote_code=True)[
            "test"
        ]
        subject_to_o1domain = {"math": "math", "phy": "math", "chem": "science"}
        ds = ds.map(
            lambda x: {
                "question": x.pop("question"),
                "solution": x.pop("gold"),
                "cot_type": subject_to_o1domain[x["subject"]],
                "source_type": "daman1209arora/jeebench/" + x["subject"],
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class AGIEvalLoader(HFLoad):
    def load_fn(self):
        conf_to_o1domain = {
            "sat_en": "english",
            "sat_math": "math",
            "lsat_ar": "english",
            "lsat_lr": "english",
            "lsat_rc": "english",
            "logiqa": "math",
        }
        confs = ["sat_en", "sat_math", "lsat_ar", "lsat_lr", "lsat_rc", "logiqa"]
        # Some have empty passages so strip in that case; 'options' field needed even for sat_math as sometimes "Which of the following" in question
        ds_with_passage = [
            datasets.load_dataset("baber/agieval", c, trust_remote_code=True).map(
                lambda x: {
                    "question": (x.pop("passage") + "\n\n").strip()
                    + x.pop("question")
                    + "\n"
                    + "\n".join(x.pop("options")),
                    "solution": x.pop("solution"),
                    "cot_type": conf_to_o1domain[c],
                    "source_type": "baber/agieval/" + c,
                    "metadata": str(x),
                }
            )
            for c in confs
        ]

        confs = ["aqua_rat"]
        ds_no_passage = [
            datasets.load_dataset("baber/agieval", c, trust_remote_code=True).map(
                lambda x: {
                    "question": x.pop("question"),
                    "solution": x.pop("solution"),
                    "cot_type": "math",
                    "source_type": "baber/agieval/" + c,
                    "metadata": str(x),
                }
            )
            for c in confs
        ]

        # Only take most difficult questions; 'options' field not really needed
        ds = (
            datasets.load_dataset(
                "baber/agieval", "math_agieval", trust_remote_code=True
            )
            .filter(lambda x: x["level"] == 5)
            .map(
                lambda x: {
                    "question": x.pop("question"),
                    "solution": x.pop("solution"),
                    "cot_type": "math",
                    "source_type": "baber/agieval/math_agieval",
                    "metadata": str(x),
                }
            )
        )

        ds = [
            datasets.concatenate_datasets([d["test"], d["few_shot"]])
            for d in ds_with_passage + ds_no_passage + [ds]
        ]
        ds = [
            d.remove_columns(
                [c for c in d.column_names if c not in constants.DS_COLUMNS]
            )
            for d in ds
        ]
        ds = datasets.concatenate_datasets(ds)
        return ds


class StatsQualLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("simplescaling/s1-prob", trust_remote_code=True)[
            "train"
        ]
        ds = ds.map(
            lambda x: {
                "question": x.pop("question"),
                "solution": x.pop("solution"),
                "cot_type": "math",
                "source_type": "simplescaling/s1-prob",
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class GPQALoader(HFLoad):
    def load_fn(self):
        gpqa_to_o1domain = {
            "Chemistry": "science",
            "Biology": "science",
            "Physics": "science",
        }
        ds = datasets.load_dataset(
            "Idavidrein/gpqa", "gpqa_extended", trust_remote_code=True
        )["train"]
        # Filter against diamond
        ds_diamond = datasets.load_dataset(
            "Idavidrein/gpqa", "gpqa_diamond", trust_remote_code=True
        )["train"]
        ds = ds.filter(lambda x: x["Question"] not in ds_diamond["Question"])
        ds = ds.map(
            lambda x: {
                "question": x.pop("Question"),
                "solution": x.pop("Explanation"),
                "cot_type": gpqa_to_o1domain[x["High-level domain"]],
                "source_type": "Idavidrein/gpqa",
                "metadata": str(x),
            }
        )

        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class XWordLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("0xharib/xword1", trust_remote_code=True)["train"]

        # Use slightly different format, e.g. would need to Rename instruction, input, output -> cl
        # ds2 = datasets.load_dataset("0xharib/xword2", trust_remote_code=True)['train']
        # ds3 = datasets.load_dataset("0xharib/xword3", trust_remote_code=True)['train']
        # ds = datasets.concatenate_datasets([ds1, ds2, ds3])

        instruction = "Solve the crossword puzzle. You are presented with a clue as input and the number of letters in brackets."
        ds = ds.map(
            lambda x: {
                "question": instruction
                + "\n\n"
                + x.pop("input").split("### Clue: ")[1],
                "solution": x.pop("output"),
                "cot_type": "crossword",
                "source_type": "0xharib/xword1",
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class USACOLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("codegenning/usacobench_formatted")["test"]
        ds = ds.map(
            lambda x: {
                "question": x.pop("question").strip(),
                "solution": None,
                "cot_type": "coding",
                "source_type": "codegenning/usacobench_formatted",
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class QuantLoader(HFLoad):
    def load_fn(self):
        ds = datasets.load_dataset("simplescaling/s1-teasers")["train"]
        ds = ds.map(
            lambda x: {
                "question": x.pop("Question").strip(),
                "solution": x.pop("Answer"),
                "cot_type": "math",
                "source_type": "qfq/quant",
                "metadata": str(x),
            }
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in constants.DS_COLUMNS]
        )
        return ds


class LiveCodeBenchLoader(HFLoad):
    def load_fn(self):
        versions = ["release_v1", "release_v2", "release_v3"]
        datasets_list = []
        for version in versions:
            ds = datasets.load_dataset(
                "livecodebench/code_generation_lite",
                version_tag=version,
                trust_remote_code=True,
            )["test"]
            ds = ds.map(
                lambda x: {
                    "question": x.pop("question_content").strip(),
                    "solution": None,
                    "cot_type": "coding",
                    "source_type": f"LiveCodeBench/{version}",
                    "metadata": str(x),
                }
            )
            # filter only the difficult questions
            ds = ds.filter(lambda x: x["difficulty"] == "hard")
            ds = ds.remove_columns(
                [c for c in ds.column_names if c not in constants.DS_COLUMNS]
            )
            datasets_list.append(ds)

        final_ds = datasets.concatenate_datasets(datasets_list)
        return final_ds


class LoaderFactory:
    @staticmethod
    def get_loader(
        ds_name: str,
        selection_strategy: SelectionStrategy = None,
        sampling_strategy: SamplingStrategy = None,
        n_samples: int = None,
    ):
        loaders = {
            "MATH": OpenAIMathLoader,
            "OlympicArena": OlympicArenaLoader,
            "TheoremQA": TheoremQALoader,
            "NuminaMath": NuminaMathLoader,
            "Omni-MATH": OmniMATHLoader,
            "SciEval": SciEvalLoader,
            "OlympiadBench": OlympiadBenchLoader,
            "JEEBench": JEEBenchLoader,
            "AGIEval": AGIEvalLoader,
            "StatsQual": StatsQualLoader,
            "GPQA": GPQALoader,
            "XWord": XWordLoader,
            "USACO": USACOLoader,
            "Quant": QuantLoader,
            "LiveCodeBench": LiveCodeBenchLoader,
        }
        if ds_name not in loaders:
            raise ValueError(f"Unsupported dataset: {ds_name}")
        return loaders[ds_name](selection_strategy, sampling_strategy, n_samples)
