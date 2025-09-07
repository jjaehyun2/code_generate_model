import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# 1. 환경설정
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 2. 경로 설정
MODEL_PATH = "./finetuned_model/finetuned"  # 학습 완료된 체크포인트 경로
DATA_PATH = "./pytorch_code_comment_pairs.json"  # 학습에 쓴 데이터
EVAL_OUTPUT = "./eval_results"                   # 평가 로그 저장 폴더

# 3. 데이터 로딩 및 전처리
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def format_sft(example):
    return {
        "instruction": example["comment"],
        "input": "",
        "output": example["code"].strip()
    }
sft_dataset = dataset.map(format_sft)

def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", "").strip():
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}

test_dataset = sft_dataset.map(format_text)

# 4. 토크나이저 및 토크나이징
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

tokenized_test = test_dataset.map(
    tokenize_function, batched=True, remove_columns=test_dataset.column_names
)

# 5. 모델 불러오기 및 평가파라미터
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

eval_args = TrainingArguments(
    output_dir=EVAL_OUTPUT,
    per_device_eval_batch_size=2
)

# 7. 트레이너 준비 및 평가
trainer = SFTTrainer(
    model=model,
    args=eval_args,
    train_dataset=tokenized_test,
    eval_dataset=tokenized_test,
)

results = trainer.evaluate()
print("평가 결과:", results)
