import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import psutil
import os
import copy
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

class PostTrainingPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pruned_model = None
    
    def apply_magnitude_pruning(self, amount=0.3):
        self.pruned_model = copy.deepcopy(self.model)
        for name, module in self.pruned_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        return self.pruned_model

    def calculate_sparsity(self, model):
        total_params, zero_params = 0, 0
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        sparsity = zero_params / total_params * 100
        return sparsity

    def get_model_size(self, model):
        """모델 크기 측정 (임시 저장 파일 크기)"""
        ts=int(time.time()*1000)
        temp_path = "temp_model_{ts}.pt"
        torch.save(model, temp_path)
        size_mb = os.path.getsize(temp_path) / 1e6
        Path(temp_path).unlink(missing_ok=True)  
        return size_mb

    def benchmark_inference(self, model, test_prompts, num_runs=5):
        model.eval()
        latencies = []
        memory_usage = []
        with torch.no_grad():
            for i in range(num_runs):
                prompt = test_prompts[i % len(test_prompts)]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    start_memory = torch.cuda.memory_allocated() / 1e6
                else:
                    start_memory = psutil.Process().memory_info().rss / 1e6
                start_time = time.time()

                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                end_time = time.time()
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated() / 1e6
                else:
                    end_memory = psutil.Process().memory_info().rss / 1e6

                latencies.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
        return {
            'avg_latency': sum(latencies) / len(latencies),
            'avg_memory': sum(memory_usage) / len(memory_usage),
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        }

    def compare_models(self, original_model, pruned_model, test_prompts):
        print("\n===== 모델 성능 비교 분석 =====")
        original_size = self.get_model_size(original_model)
        optimized_size = self.get_model_size(pruned_model)
        
        print(f"\n📊 모델 크기 비교:")
        print(f"   원본 모델: {original_size:.1f} MB")
        print(f"   최적화 모델: {optimized_size:.1f} MB")
        print(f"   크기 감소: {((original_size - optimized_size) / original_size * 100):.1f}%")

        original_perf = self.benchmark_inference(original_model, test_prompts)
        pruned_perf = self.benchmark_inference(pruned_model, test_prompts)
        print(f"\n원본 모델: 평균 추론 시간 {original_perf['avg_latency']:.3f}초, 평균 메모리 사용 {original_perf['avg_memory']:.1f}MB")
        print(f"가지치기 모델: 평균 추론 시간 {pruned_perf['avg_latency']:.3f}초, 평균 메모리 사용 {pruned_perf['avg_memory']:.1f}MB")

        speed_improvement = 100 * (original_perf['avg_latency'] - pruned_perf['avg_latency']) / original_perf['avg_latency']
        if original_perf['avg_memory'] > 0:
            memory_reduction = 100 * (original_perf['avg_memory'] - pruned_perf['avg_memory']) / original_perf['avg_memory']
        else:
            memory_reduction = 0.0

        print(f"\n 성능 개선 효과:")
        print(f"   추론 속도 개선: {speed_improvement:.1f}%")
        print(f"   메모리 사용량 감소: {memory_reduction:.1f}%")

        sparsity = self.calculate_sparsity(pruned_model)
        print(f"모델 희소성(파라미터 0 비율): {sparsity:.1f}%")

        return {
            'speed_improvement': speed_improvement,
            'memory_reduction': memory_reduction,
            'sparsity': sparsity
        }


def main():
    model_path = "./finetuned_model/finetuned_V1_quantized_pruned"  
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    pruner = PostTrainingPruner(model, tokenizer)

    pruned_model = pruner.apply_magnitude_pruning(amount=0.3)

    test_prompts = [
        "Generate a Python function to calculate fibonacci numbers:",
        "Create a function to sort a list of integers:"
    ]

    pruner.compare_models(model, pruned_model, test_prompts)

    save_dir = "./finetuned_model/finetuned_V2_quantized_pruned"
    pruned_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"가지치기된 모델 저장 완료: {save_dir}")

if __name__ == "__main__":
    main()
