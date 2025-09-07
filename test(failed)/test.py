from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 및 토크나이저 로드
model_path = "./finetuend_model/finetuned"  # 학습 완료된 모델 경로
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# 테스트 함수
def generate_code(instruction, input_text=""):
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[1].strip()

# 테스트 예제
test_cases = [
    {
        "instruction": "Write a Python function to find the factorial of a number.",
        "input": ""
    },
    {
        "instruction": "Explain the following code:",
        "input": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]"
    }
    
]

# 테스트 실행
print("Testing fine-tuned model...")
for i, test in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    print(f"Instruction: {test['instruction']}")
    if test['input']:
        print(f"Input: {test['input']}")
    
    response = generate_code(test['instruction'], test['input'])
    print("\nModel Response:")
    print(response)
    print("-" * 50)

print("Testing complete!")