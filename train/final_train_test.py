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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_path = "./finetuned_model/finetuned_V1_quantized_pruned" 
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

for param in model.parameters():
    param.requires_grad = True


tokenizer.pad_token = tokenizer.eos_token

def load_and_preprocess_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    processed_data = []
    for item in data:

        input_text = f"### Instruction:\n{item['instruction']}\n\n### Output:\n{item['output']}"
        processed_data.append({
            "text": input_text
        })
    
    return processed_data


def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=128, 
        padding='max_length',
        return_tensors='pt'
    )


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

dataset_path = "./dataset/train_data.json"  
dataset = load_and_preprocess_dataset(dataset_path)


from datasets import Dataset
dataset = Dataset.from_list(dataset)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


trainer.train()


trainer.save_model("./finetuned_model/finetuned_V2")