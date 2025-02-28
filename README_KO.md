# S1 clone
<p align="left">
      &nbsp한국어&nbsp ｜ <a href="README.md">English</a>&nbsp
</p>

## 사용법

### Install
- `uv`가 설치되어 있는지 확인하세요. 설치되지 않았다면 먼저 설치하세요:
    
    ```bash
    pip install uv
    ```
    
- 종속성 설치:
    
    ```bash
    uv sync
    ```
### 가상 환경 활성화
```bash
source .venv/bin/activate
```

### 환경 변수 설정

`.env` 파일을 생성하고 아래 값을 입력하세요.

```plaintext
# Hugging Face 데이터셋 관리를 위한 자격 증명
HF_TOKEN="your_huggingface_token"
HF_USERNAME="your_huggingface_username"

# featuring 및 채점용 API 키
ANTHROPIC_API_KEY="your_claude_api_key"
```

## 데이터셋 준비

### 1. 데이터 수집
S1 논문에서 초기 59K 데이터셋을 생성하고 Hugging Face에 업로드합니다.
```python
python data/collect_data.py
```

### 2. GPQA 데이터 포맷 변환
GPQA 데이터셋의 질문과 솔루션을 다중 선택 평가 형식으로 변환합니다.
```python
python data/fix_gpqa.py
```

### 3. AIME 데이터 추가
AIME 관련 데이터 증강을 포함합니다.
```python
python data/add_aime.py
```

### 4. Generate Reasoning(thinking trajectory)
**DeepSeek-R1-Distill-Qwen-32B(reasoning model)**를 사용하여 reasoning 레이블을 생성합니다.

**권장 설정**: 최소 **4x H100 GPU**. 이 설정에서 **2048개 샘플**을 처리하는 데 약 1시간이 소요됩니다.

`data/generate_reasoning.py`의 `tensor_parallel_size`를 사용 중인 GPU 개수에 맞게 조정하세요.

기본 설정으로 추론을 생성하려면:
```bash
bash scripts/generate_reasoning.sh
```

설정을 사용자 정의하려면:
```bash
python data/generate_reasoning.py --model_name={model_name} --max-model_len={max_model_len} ... 
```

### 5. Difficulty Classification(난이도 분류)
사전 학습된 언어 모델을 사용하여 질문의 난이도를 분류합니다.

**사용법:**
```bash
bash scripts/difficulty_classify.sh
```

### 6. featurization
질문의 도메인을 분류하고 모델 응답을 평가한 후 결과를 업로드합니다.

`.env` 파일에 `ANTHROPIC_API_KEY`가 필요합니다.

**사용법:**
```bash
python data/featurization.py
```

#### 피처링 단계:

1. **도메인 분류** - `claude-3-5-sonnet-20241022`를 사용하여 질문을 분류합니다.
2. **채점** - AI 생성 답변이 올바른지 여부를 `data/grading.txt`의 채점 모델을 통해 판별합니다.
3. **채점 결과 업로드** - 채점 정보를 데이터셋에 업데이트합니다.
4. **토큰 길이 업로드** - 모델 응답의 토큰 길이를 계산하여 업데이트합니다.

### 7.필터링 및 샘플링
**1K(1,000)개 샘플**의 정제된 데이터셋을 생성합니다.

**사용법:**
```bash
python data/filter.py
```

#### 필터링 프로세스:
- 누락된 값 제거
- 특정 문자열 패턴(weired pattern)을 포함하는 질문 제거
- Qwen 모델(비추론 모델)이 틀린 질문만 유지
- 도메인 다양성 보장
- power-law 기반 길이 샘플링 적용
- 최종 데이터셋을 로컬에 저장하거나 Hugging Face에 업로드

## 학습
(TBD - 학습 명령어 추가 예정)

## 평가
(TBD - 평가 방법 추가 예정)

### 인용

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