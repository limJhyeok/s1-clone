# S1 clone


## Usage

### Install

- OS: Ubuntu 22.04
- Python: 3.11
- Nvidia driver version: 12.4

```bash
pip install -r requirements.txt
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
