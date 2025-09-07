import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set GPU devices
# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델과 토크나이저 로드
model_path = "./finetuned_model/finetuned_V1_quantized_pruned"  # 모델 경로
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 모델의 모든 파라미터에 대해 그래디언트 계산 활성화
for param in model.parameters():
    param.requires_grad = True

# 토크나이저에 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token
# 데이터셋 로드 및 전처리 함수
def load_and_preprocess_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터셋 형식에 맞게 변환
    processed_data = []
    for item in data:
        # instruction과 output을 직접 활용
        input_text = f"### Instruction:\n{item['instruction']}\n\n### Output:\n{item['output']}"
        processed_data.append({
            "text": input_text
        })
    
    return processed_data

# 데이터셋 토큰화 함수
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=128, 
        padding='max_length',
        return_tensors='pt'
    )

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100
)

# 데이터셋 준비
dataset_path = "./dataset/train_data.json"  # 실제 데이터셋 경로로 대체
dataset = load_and_preprocess_dataset(dataset_path)

# 데이터셋을 Hugging Face Dataset으로 변환
from datasets import Dataset
dataset = Dataset.from_list(dataset)

# 데이터셋 토큰화
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 데이터 콜레이터 준비
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 모델 학습
trainer.train()

# 모델 저장
trainer.save_model("./finetuned_model/finetuned_V2")