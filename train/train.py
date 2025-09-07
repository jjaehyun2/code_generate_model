import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6" 

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

dataset = load_dataset("json", data_files="./dataset/train_data.json")
train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = train_val["test"]

def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}



model_id = "jack0503/code_generate_explain"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
train_dataset = train_dataset.map(format_text)
val_dataset = val_dataset.map(format_text)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)



model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={'': 0},
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./qwen-code-assistant",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting training...")
trainer.train()
print("Saving model...")
trainer.save_model("./qwen-code-assistant-largemodel")
print("Training complete!")