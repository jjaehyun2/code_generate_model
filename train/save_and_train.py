import os
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer
from sklearn.model_selection import train_test_split


os.environ["WANDB_MODE"] = "offline"

# CUDA 디바이스 설정 (필요에 따라 변경)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 1) 경로 및 변수 설정
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"  # HuggingFace 허브 베이스 모델
ADAPTER_REPO = "jack0503/my-hf-model"    # 허브에 업로드한 어댑터 repo ID
DATA_PATH = "./dataset/pandas_code_comment_pairs.json"  # JSON 데이터 경로 (Drive 기준)
SAVE_PATH = "./final_train"                 # 학습 결과 저장 위치

# 2) JSON 데이터 로드
# 2) JSON 데이터 로드
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


def split_dataset(data):
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 20% test
    train_data, val_data = train_test_split(train_val_data, test_size=1/8, random_state=42)  # 10% val (총 80%에서 1/8)
    return train_data, val_data, test_data


train_data, val_data, test_data = split_dataset(data)

# Dataset 객체 생성
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)


def format_sft(example):
    return {
        "instruction": example["comment"],
        "input": "",
        "output": example["code"].strip()
    }


def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", "").strip():
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}


# 각 데이터셋에 format_sft와 format_text 적용
train_dataset = train_dataset.map(format_sft).map(format_text)
val_dataset = val_dataset.map(format_sft).map(format_text)
test_dataset = test_dataset.map(format_sft).map(format_text)


# 토크나이저 준비
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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


# 각 데이터셋 토큰화
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", use_cache=False)
base_model.gradient_checkpointing_enable()
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, use_auth_token=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=100,
    save_steps=1000,
    fp16=False,
    bf16=False,
    label_names=[],
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)


print("SFT 파인튜닝 시작!")
trainer.train()  # 필요시 재개 체크포인트 지정


print("저장 중...")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"학습 완료! 저장위치: {SAVE_PATH}")
