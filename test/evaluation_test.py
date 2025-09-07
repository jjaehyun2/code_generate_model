import os
import sys
import tempfile
import subprocess
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from safetensors.torch import load_file


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# 1) 베이스 모델 경로 (온라인 허깅페이스 허브 또는 로컬 전체 모델)
base_model_dir = "Qwen/Qwen2.5-3B-Instruct"  # 예시

# 2) 어댑터 경로
adapter_dir = "/workspace/test/code_chat/final_train"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True, trust_remote_code=True)  # 토크나이저는 adapter 디렉토리에서 충분할 수 있음

# 베이스 모델 로드
model = AutoModelForCausalLM.from_pretrained(base_model_dir)

# 어댑터 가중치 병합 (safetensors)

adapter_weights = load_file(f"{adapter_dir}/adapter_model.safetensors")

# HumanEval 로드
data = load_dataset("openai_humaneval")["test"]

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 이후만 코드 후보로 추출
    return full[len(prompt):].strip()

def extract_function_only(generated_code):
    """
    'def ...'부터 시작하는 첫 함수 선언 ~ 끝까지 만 리턴.
    추가 설명, 주석, warning 등 제거
    """
    lines = generated_code.splitlines()
    start, end = -1, len(lines)
    for idx, line in enumerate(lines):
        if line.strip().startswith("def "):
            start = idx
            break
    if start == -1:
        return ""  # 함수 없음
    # 끝을 찾아 전체 함수+바로 아래 코드 (보통 doctest 등)까지 포함
    return "\n".join(lines[start:])

def check_with_subprocess(candidate_code, test_code):
    """
    후보코드와 테스트코드를 합쳐 임시 py 파일로 만들고,
    python -c "실행"으로 판정(통과: 1 / 실패: 0)
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(candidate_code)
        f.write("\n\n")
        f.write(test_code)
        temp_path = f.name
    try:
        # 10초 제한, 실패는 실패로 판정
        result = subprocess.run(
            [sys.executable, temp_path],
            timeout=10,
            capture_output=True,
            text=True,
        )
        success = result.returncode == 0
    except Exception as e:
        success = False
    os.remove(temp_path)
    return success

num_total, num_pass = 0, 0

for idx, problem in enumerate(data):
    prompt = problem["prompt"]
    test_code = problem["test"]
    gen_code = generate_code(prompt)
    func_code = extract_function_only(gen_code)
    result = check_with_subprocess(func_code, test_code)
    num_total += 1
    num_pass += int(result)
    print(f"[{idx}] {'PASS' if result else 'FAIL'}")
    print(func_code)
    print("="*40)

print(f"HumanEval 채점 결과: {num_pass}/{num_total} 통과 (accuracy={num_pass/num_total:.3%})")
