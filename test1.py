import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import csv
import os
from statistics import mean
import pynvml
import subprocess
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 설정 (필요에 따라 조정)
# ==========================
# 설정
# ==========================
MODEL_PATHS = {
    "model_a": "Qwen/Qwen2.5-3B-Instruct",
    "model_b": "jack0503/code_generate_explain",
}

PROMPT_FILE = "python_prompts_100.csv"  # 한 줄당 하나의 코드 생성 프롬프트
RESULT_DIR = "./benchmark_results"
REPEATS = 5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# GPU 모니터링 초기화
# ==========================
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats():
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024**2)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
    return mem, util

# ==========================
# CodeBLEU 계산 (간단 버전)
# ==========================
def compute_codebleu(pred, ref):
    # 실험용 간이 점수 (실전에서는 Salesforce CodeBLEU 패키지 사용)
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    overlap = len(set(pred_tokens) & set(ref_tokens))
    return overlap / max(len(set(ref_tokens)), 1)

# ==========================
# 데이터 로드
# ==========================
if not os.path.exists(PROMPT_FILE):
    raise FileNotFoundError(f"{PROMPT_FILE} not found.")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================
# 실행
# ==========================
for model_id, model_path in MODEL_PATHS.items():
    print(f"\n=== Loading model: {model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    result_file = os.path.join(RESULT_DIR, f"{model_id}_results.csv")
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "prompt_id", "run_idx", "prompt_len_tokens",
            "FTL_ms", "TTC_ms", "tokens_generated", "TPS",
            "gpu_mem_MB", "gpu_util_pct", "codebleu"
        ])

        # 워밍업
        print("Warming up...")
        for _ in range(5):
            _ = model.generate(
                **tokenizer("warmup", return_tensors="pt").to(DEVICE),
                max_new_tokens=16
            )

        # 본 실험
        for prompt_id, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            prompt_len = inputs["input_ids"].shape[1]

            # 참조 정답이 있다면 품질 측정 가능 (여기서는 빈 값)
            reference = ""  

            for r in range(REPEATS):
                torch.cuda.synchronize()
                start_time = time.time()
                mem_before, util_before = get_gpu_stats()

                # 첫 토큰 Latency 측정
                first_token_time = None

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P
                    )

                torch.cuda.synchronize()
                end_time = time.time()

                # 토큰 수, 속도 계산
                total_tokens = output_ids.shape[1] - prompt_len
                ttc_ms = (end_time - start_time) * 1000
                tps = total_tokens / (ttc_ms / 1000) if total_tokens > 0 else 0

                mem_after, util_after = get_gpu_stats()

                # 품질 점수 (여기서는 참조 없음)
                generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
                codebleu_score = compute_codebleu(generated_text, reference)

                writer.writerow([
                    prompt_id, r, prompt_len,
                    None, round(ttc_ms, 2), total_tokens, round(tps, 2),
                    mem_after, util_after, round(codebleu_score, 4)
                ])

    print(f"Results saved to {result_file}")

pynvml.nvmlShutdown()
