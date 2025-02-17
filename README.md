# S1 clone


## Usage

### Install

```bash
# install poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# install requirements
poetry install --no-root

# apply pre-commit hooks
poetry run pre-commit install
```

#### 🔧 Handling vLLM Dependency Issue

I tried to install `vllm` using poetry. But it seems poetry can't resolve the dependency problem.

⚠ Related Issue: [vLLM #8851](https://github.com/vllm-project/vllm/issues/8851)


So after installing libraries except for vLLM using poetry and then install vLLM using requirements.txt. It will update other dependencies as well.

```bash
pip install -r requirements.txt
```

### Activate Virtual Environment
```bash
eval $(poetry env activate)
```


### .env file setting

fill in the `HF_TOKEN`, `HF_USERNAME` in .env file

```plain text
# for download dataset from huggingface and upload to huggingface.
HF_TOKEN=""

# for upload dataset to huggingface
HF_USERNAME = ""

```
- HF_TOKEN: Hugging Face Token
- HF_USERNAME: Hugging Face User Name

## Dataset
```python
python data/collect_data.py
```
It will make the first **59k data** in s1 and then upload to your Hugging Face Dataset repository if you properly set the `HF_TOKEN` and `HF_USERNAME`

### GPQA Formatting

#### Overview

The `data/fix_gpqa.py` script reformats the dataset by modifying the structure of the `question` and `solution` fields. This transformation standardizes the dataset to include answer choices explicitly and appends the correct answer to the solution.

To apply this transformation, simply run:
```python
python data/fix_gpqa.py
```
This will process the dataset and update it to the new structured format, making it more suitable for multiple-choice evaluations and automated processing.


### Add AIME data

```python
python data/add_aime.py
```

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
