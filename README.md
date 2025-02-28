# S1 clone
<p align="left">
    <a href="README_KO.md">한국어</a>&nbsp ｜ &nbspEnglish&nbsp
</p>

## Usage

### Install
- Ensure you have `uv` installed. If not, install it first:
    
    ```bash
    pip install uv
    ```
    
- Install dependencies:
    
    ```bash
    uv sync
    ```
### Activate Virtual Environment
```bash
source .venv/bin/activate
```


### Environment Variables Setup

fill in the `HF_TOKEN`, `HF_USERNAME` in .env file

```plain text
# Hugging Face credentials for dataset management
HF_TOKEN="your_huggingface_token"
HF_USERNAME="your_huggingface_username"

# API Key for featurization and grading
ANTHROPIC_API_KEY="your_claude_api_key"
```

## Dataset Preparation

### 1. Collect Data
enerates the initial 59K dataset from the S1 paper and uploads it to Hugging Face.
```python
python data/collect_data.py
```

### 2. Format GPQA Data
Reformats dataset questions and solutions for multiple-choice evaluations.
```python
python data/fix_gpqa.py
```

### 3. Add AIME data
Includes AIME-related dataset augmentation.
```python
python data/add_aime.py
```

### 4. Generate Reasoning(thinking trajectory)
Generates reasoning labels using **DeepSeek-R1-Distill-Qwen-32B**.

**Recommended Setup**: At least **4x H100 GPUs**. With this setup, processing **2048 samples** takes approximately 1 hour

you should change `tensor_parallel_size` in `data/generate_reasoning.py` to your numbers of GPU machines.

To generate reasoning with default settings:
```bash
bash scripts/generate_reasoning.sh
```

To customize settings:
```bash
python data/generate_reasoning.py --model_name={model_name} --max-model_len={max_model_len} ... 
```

### 5.Difficulty Classification
classify the difficulty of questions using a pretrained language model.

**Usage:**
```bash
bash scripts/difficulty_classify.sh
```

### 6.featurization
Classifies question domains, evaluates model responses, and uploads results.

Requires `ANTHROPIC_API_KEY` in `.env` file.

**Usage:**
```bash
python data/featurization.py
```
#### Featurization Steps:

1. **Domain Classification** - Uses `claude-3-5-sonnet-20241022` to classify questions.
2. **Grading** - Determines if AI-generated answers are correct using a grading model (`data/grading.txt`).
3. **Upload Grading Results** - Updates the dataset with grading information.
4. **Upload Token Length** - Computes and updates token lengths for model responses.

### 7. filtering & Sampling
Prepares a refined dataset (up to 1K samples) for reasoning-based tasks.

**Usage:**
```bash
python filter.py
```

#### Filtering Process:

- Removes missing values.
- Eliminates questions with undesired text patterns.
- Retains only questions incorrectly answered by Qwen models(non-reasoning models).
- Ensures domain diversity.
- Applies power-law length sampling.
- Saves the final dataset locally or uploads it to Hugging Face.
## Train
(TBD - Training commands to be added)
## Eval
(TBD - Evaluation steps to be included)
### Citation

```bibtex
@misc{muennighoff2025s1simpletesttimescaling,
      title={s1: Simple test-time scaling},
      author={Niklas Muennighoff and Zitong Yang and Weijia Shi and Xiang Lisa Li and Li Fei-Fei and Hannaneh Hajishirzi and Luke Zettlemoyer and Percy Liang and Emmanuel Candès and Tatsunori Hashimoto},
      year={2025},
      eprint={2501.19393},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.19393},
}
```
