# Code Generate Model

이 프로젝트는 **코드 생성(Code Generation)** 및 **코드 설명(Code Explanation)**,  
그리고 **모델 최적화 및 평가**를 위한 다양한 파이프라인을 제공합니다.  

Hugging Face Transformers, Datasets 라이브러리를 기반으로 하며,  
경량화 및 성능 향상을 위해 LoRA, 양자화(Quantization), 프루닝(Pruning) 등의 기법을 지원했습니다.
---

## 🚀 주요 기능
- 코드 생성 및 코드 설명
- LoRA, PEFT 기반 파인튜닝
- 양자화 및 프루닝을 통한 모델 경량화
- HumanEval 기반 성능 평가
- BLEU, BERTScore 등 다양한 평가 지표 제공

---

## 📂 프로젝트 구조

### 1. `train/`
- **train.py**  
  - JSON 데이터셋을 텍스트 포맷으로 변환 후, 사전학습된 모델(`AutoModelForCausalLM`)을 LoRA 등으로 파인튜닝  
  - 4bit 양자화(`BitsAndBytesConfig`) 등 다양한 경량화 옵션 제공  

- **save_and_train.py**  
  - 데이터셋을 학습/검증/테스트로 분할  
  - `SFTTrainer`를 활용한 파인튜닝 및 모델 저장  
  - PEFT(파라미터 효율적 파인튜닝)와 Hugging Face Hub 연동 지원  

- **final_train_test.py**  
  - 파인튜닝 및 경량화/프루닝된 모델을 불러와 추가 학습 및 테스트 수행  
  - 데이터 전처리, 토크나이저 설정, CUDA 환경 지원  

---

### 2. `upgrade/`
- **dynamic_quantization.py**  
  - 사전학습된 모델에 동적 양자화(Dynamic Quantization) 적용  
  - 메모리 사용량 절감 및 추론 속도 향상  
  - 모델 로딩 → 토크나이저 설정 → 최적화 적용 및 저장  

- **post_training_pruning.py**  
  - 학습 후 프루닝(Post-Training Pruning)을 통해 모델 경량화  
  - L1 기반 Unstructured Pruning  
  - 희소도 계산 및 프루닝된 모델 반환  

---

### 3. `test(failed)/` 
- **evaluation_test.py / evaluation_test2.py**  
  - HumanEval 데이터셋 기반 코드 생성 성능 평가  
  - 어댑터 모델과 베이스 모델 결합, `safetensors` 지원  

- **test.py / test_for_new.py**  
  - 파인튜닝된 모델 불러오기 및 코드 생성 테스트  
  - 입력 프롬프트 기반 코드 생성 함수 제공  

- **test_code_explain.py**  
  - 코드 설명 생성 기능 제공  
  - BLEU, BERTScore 등 다양한 지표를 활용한 성능 평가  

---

## 추후 작업
- 코드 평가는 더 복합적인 기능을 추가하여 별도의 작업 진행
- humaneval이 아닌 정답 데이터셋을 활용한 다방면 유사도 평가 예정
- 추가적인 논문 리서치 예정

