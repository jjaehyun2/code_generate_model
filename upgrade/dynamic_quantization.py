import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import psutil
import gc
from pathlib import Path
if os.path.exists("temp_model.pt"):
    os.remove("temp_model.pt") 

os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

class ModelOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.original_model = None
        self.optimized_model = None
        self.tokenizer = None
        
    def load_model(self):
        """원본 모델 로드"""
        print("원본 모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.original_model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float32)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ 원본 모델 로딩 완료")
    
    def apply_optimization(self):
        if torch.cuda.is_available():
            self.optimized_model = self.original_model.to(torch.bfloat16)
        else:
            self.optimized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        return self.optimized_model
    
    def get_model_size(self, model):
        """모델 크기 측정 (임시 저장 파일 크기)"""
        ts=int(time.time()*1000)
        temp_path = "temp_model_{ts}.pt"
        torch.save(model, temp_path)
        size_mb = os.path.getsize(temp_path) / 1e6
        Path(temp_path).unlink(missing_ok=True)  # 파일 삭제
        return size_mb
    
    def benchmark_inference(self, model, test_prompts, num_runs=5):
        """추론 벤치마크 (평균 시간, 메모리 사용 측정)"""
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
                
                if model.dtype == torch.bfloat16:
                    inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
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
    
    def compare_models(self, test_prompts):
        """원본 모델과 최적화 모델 성능 비교 및 로그 출력"""
        print("\n" + "="*60)
        print("모델 성능 비교 분석")
        print("="*60)
        
        original_size = self.get_model_size(self.original_model)
        optimized_size = self.get_model_size(self.optimized_model)
        
        print(f"\n📊 모델 크기 비교:")
        print(f"   원본 모델: {original_size:.1f} MB")
        print(f"   최적화 모델: {optimized_size:.1f} MB")
        print(f"   크기 감소: {((original_size - optimized_size) / original_size * 100):.1f}%")
        
        print(f"\n 추론 성능 비교:")
        print("   원본 모델 벤치마킹...")
        original_perf = self.benchmark_inference(self.original_model, test_prompts)
        
        print("   최적화 모델 벤치마킹...")
        optimized_perf = self.benchmark_inference(self.optimized_model, test_prompts)
        
        print(f"\n   원본 모델:")
        print(f"     평균 추론 시간: {original_perf['avg_latency']:.3f}초")
        print(f"     평균 메모리 사용: {original_perf['avg_memory']:.1f}MB")
        
        print(f"\n   최적화 모델:")
        print(f"     평균 추론 시간: {optimized_perf['avg_latency']:.3f}초")
        print(f"     평균 메모리 사용: {optimized_perf['avg_memory']:.1f}MB")
        
        speed_improvement = (original_perf['avg_latency'] - optimized_perf['avg_latency']) / original_perf['avg_latency'] * 100
        if original_perf['avg_memory'] > 0:
            memory_reduction = (original_perf['avg_memory'] - optimized_perf['avg_memory']) / original_perf['avg_memory'] * 100
        else:
            memory_reduction = 0
        
        print(f"\n 성능 개선 효과:")
        print(f"   추론 속도 개선: {speed_improvement:.1f}%")
        print(f"   메모리 사용량 감소: {memory_reduction:.1f}%")
        
        return {
            'size_reduction': ((original_size - optimized_size) / original_size * 100),
            'speed_improvement': speed_improvement,
            'memory_reduction': memory_reduction
        }
    
    def save_optimized_model(self, save_dir: str="optimized_model"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.optimized_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"최적화된 모델을 {save_dir}에 저장했습니다.")
    
    def load_optimized_model(self, load_dir: str="optimized_model"):
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        return model, tokenizer

def main():
    model_path = "./finetuned_model/finetuned_V1_quantized_pruned"  
    optimizer = ModelOptimizer(model_path)
    optimizer.load_model()
    

    optimizer.apply_optimization()
    
    test_prompts = [
        "Generate a Python function to calculate fibonacci numbers:",
        "Create a function to sort a list of integers:",
        "Write a Python class for a binary tree:",
        "Implement a function to check if a string is palindrome:",
        "Generate code for reading a CSV file:"
    ]
    
    results = optimizer.compare_models(test_prompts)
    
    optimizer.save_optimized_model("finetuned_model/finetuned_V2_optimized")
    
    print(f"\n최종 최적화 결과:")
    print(f" 모델 크기: {results['size_reduction']:.1f}% 감소")
    print(f" 추론 속도: {results['speed_improvement']:.1f}% 향상")
    print(f" 메모리 사용: {results['memory_reduction']:.1f}% 절약")

if __name__ == "__main__":
    main()
