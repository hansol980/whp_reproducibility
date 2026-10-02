# RTX 4080 단일 GPU 환경에서의 WHP 재현 실험

## 1. 프로젝트 개요

이 프로젝트는 아래 공식 저장소를 기반으로 구성했습니다.

- 원본 저장소: [ucsb-nlp-chang/causal_unlearn](https://github.com/ucsb-nlp-chang/causal_unlearn)
- 후속 논문: *Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective*
- 선행 논문: *Who's Harry Potter? Approximate Unlearning in LLMs*

원본 저장소에는 선행 논문의 WHP(Who's Harry Potter) 방법을 재현하는 베이스라인과, 이를 인과 개입 관점에서 확장한 방법이 함께 구현되어 있습니다. 다만 실험 데이터는 해리포터 소설 원문이 아니라 TOFU와 WPU 데이터셋입니다.

이 저장소는 원본 코드를 복사한 뒤, **RTX 4080 1장으로 TOFU의 WHP 베이스라인 실험을 실행할 수 있도록 수정한 버전**입니다.

## 2. 컴퓨팅 자원 제약

사용 가능한 환경은 다음과 같습니다.

- GPU: NVIDIA RTX 4080 1장
- GPU 메모리: 16GB
- 실행 환경: Vast.ai의 Linux 인스턴스
- 비용 제약: A100 또는 다중 GPU 인스턴스를 장시간 사용할 수 없음

원본 설정은 다음과 같은 고사양 환경을 전제로 합니다.

- Llama 2 7B 모델을 BF16으로 로드
- 전체 파라미터 미세조정
- DeepSpeed ZeRO-3 사용
- `torchrun`을 통한 GPU 2장 분산 학습
- TOFU 학습 배치 크기 16 사용

Llama 2 7B의 BF16 가중치만 약 14GB를 차지하며, 실제 학습에는 활성값, 그래디언트, 옵티마이저 상태와 로짓을 위한 추가 메모리가 필요합니다. 따라서 원본의 전체 파라미터 미세조정 설정은 16GB VRAM에서 실행하기 어렵습니다.

이 프로젝트에서는 동일한 WHP 손실과 학습 흐름을 유지하면서, 4-bit QLoRA 방식으로 메모리 사용량을 줄였습니다.

## 3. 코드 수정 사항

### 3.1 4-bit NF4 모델 로딩

`model_loading.py`를 추가하여 학습, teacher 생성, 평가에서 동일한 방식으로 모델을 불러오도록 통합했습니다.

```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true
```

모델 가중치는 4-bit NF4로 양자화하고 실제 계산은 BF16으로 수행합니다. GPU가 BF16을 안정적으로 지원하지 않으면 설정을 `float16`으로 변경할 수 있습니다.

### 3.2 전체 미세조정을 QLoRA로 변경

원본은 모델 전체 파라미터를 학습하지만, 수정 버전은 양자화된 선형 계층에 LoRA 어댑터를 추가하고 어댑터 파라미터만 학습합니다.

```yaml
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

4-bit 학습 결과는 전체 모델이 아니라 PEFT 어댑터 형태로 저장됩니다. 평가 코드는 저장된 어댑터를 자동으로 인식하여 기반 모델과 함께 불러옵니다.

### 3.3 단일 GPU 설정

RTX 4080 한 장에서 동작하도록 다음 설정을 적용했습니다.

```yaml
batch_size: 1
gradient_accumulation_steps: 16
num_epochs: 1
deepspeed: null
```

DeepSpeed와 다중 GPU 실행을 제거하고, 실제 배치 크기는 1로 줄였습니다. 그래디언트 누적을 16회 수행하여 원본 TOFU 설정의 유효 배치 크기 16에 가깝게 맞췄습니다.

### 3.4 WHP 베이스라인 설정 고정

다음 옵션으로 후속 논문 저장소에 포함된 WHP 베이스라인을 실행합니다.

```yaml
forget_loss: whp
whp_baseline: true
counter_fact_prompt: false
change_name_back: true
teacher:
  teacher_method: N1
```

학습 전에 `N1` 방식의 teacher 로짓을 생성하고, WHP 손실 계산에 해당 결과를 사용합니다.

### 3.5 단일 GPU 평가 지원

평가 과정도 배치 크기 1과 4-bit 로딩을 사용하도록 별도 설정을 추가했습니다. LoRA 어댑터가 저장된 경로를 전달하면 평가 코드가 기반 모델과 어댑터를 함께 로드합니다.

## 4. 주요 파일

| 파일 | 역할 |
| --- | --- |
| `model_loading.py` | 4-bit 기반 모델과 PEFT 어댑터 로딩을 통합 |
| `config/forget_tofu_4080.yaml` | RTX 4080용 WHP 학습 설정 |
| `config/eval_everything_4080.yaml` | RTX 4080용 평가 설정 |
| `teacher.py` | 4-bit 모델로 teacher 로짓 생성 |
| `forget.py` | QLoRA 준비, 단일 GPU 학습, 어댑터 저장 |
| `evaluate_util.py` | 4-bit 기반 모델 또는 LoRA 어댑터 평가 |
| `scripts/run_tofu_whp_4080.sh` | teacher 생성부터 학습과 평가까지 순차 실행 |
| `scripts/eval_tofu_4080.sh` | 저장된 체크포인트만 다시 평가 |

## 5. 재현 범위와 한계

현재 기본 설정은 TOFU의 `forget01` 분할에서 WHP 베이스라인을 실행합니다.

주의할 점은 다음과 같습니다.

- 해리포터 소설 원문을 데이터로 사용하지 않습니다.
- 선행 WHP 논문에서 사용한 reinforced model 생성 단계 전체를 별도로 재구현한 것은 아닙니다.
- 후속 논문의 공식 저장소가 제공하는 `teacher_method=N1`, `whp_baseline=true` 경로를 실행합니다.
- 원본의 전체 파라미터 미세조정 대신 QLoRA를 사용하므로, 원 논문의 수치와 직접 비교하기보다는 제한된 자원에서의 재현 및 구현 검증으로 해석해야 합니다.
- 여러 언러닝 방법을 비교할 경우 모든 방법에 동일한 QLoRA, 배치 크기, 학습 epoch 설정을 적용해야 공정한 비교가 가능합니다.

## 6. Vast.ai 환경 구축

권장 Docker 이미지는 다음과 같습니다.

```text
pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel
```

디스크는 데이터, 모델 캐시, 체크포인트를 고려하여 최소 100GB를 권장합니다.

### 6.1 저장소 이동 및 가상환경 생성

```bash
cd whp-repro

conda create -n whp python=3.10 -y
conda activate whp
```

### 6.2 패키지 설치

```bash
pip install -r requirements.txt
pip install flash-attn==2.5.6 --no-build-isolation

python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
python -c "import nltk; nltk.download('punkt')"
```

`flash-attn` 설치가 실패하면 우선 해당 설치만 건너뛰고 나머지 환경을 확인할 수 있습니다. 다만 원본 평가 코드 일부가 이를 요구할 수 있으므로 최종 평가 전에는 설치 상태를 확인해야 합니다.

### 6.3 Hugging Face 인증

Llama 2 사용 권한이 승인된 Hugging Face 계정의 토큰이 필요합니다.

```bash
huggingface-cli login
```

- 모델: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- 데이터: [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)

## 7. 전체 실험 실행

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
bash scripts/run_tofu_whp_4080.sh "$PWD/results" 0 forget01 1
```

인자의 의미는 순서대로 다음과 같습니다.

1. 결과 저장 경로
2. 사용할 GPU 번호
3. TOFU 분할 이름
4. 학습 epoch 수

기본 명령은 다음 순서로 실행됩니다.

1. `N1` teacher 로짓 생성
2. WHP 베이스라인 QLoRA 학습
3. 체크포인트 평가
4. 전체 평가 결과 집계

메모리 여유와 비용을 확인한 뒤 5 epoch 실험을 실행하려면 마지막 인자를 `5`로 변경합니다.

```bash
bash scripts/run_tofu_whp_4080.sh "$PWD/results" 0 forget01 5
```

## 8. 결과 저장 위치

기본 실행 결과는 다음 위치에 저장됩니다.

```text
results/
├── locuslab/tofu_ft_llama2-7b/whp/forget01.pkl
└── whp-baseline-forget01-ep1/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── checkpoint-*/
    └── aggregate_stat.csv
```

teacher 로짓 파일이 이미 존재하면 실행 스크립트는 해당 단계를 건너뜁니다. 학습된 어댑터만 다시 평가하려면 다음 명령을 사용합니다.

```bash
bash scripts/eval_tofu_4080.sh \
  "$PWD/results/whp-baseline-forget01-ep1" \
  forget01 \
  0
```

## 9. 현재 검증 상태

- Python 파일 문법 검사 완료
- 실행 스크립트 문법 검사 완료
- YAML 설정 파일 파싱 확인 완료
- Git diff 공백 오류 검사 완료
- 실제 CUDA 학습 및 평가는 로컬 환경에 NVIDIA GPU가 없어 Vast.ai RTX 4080 인스턴스에서 확인해야 함
