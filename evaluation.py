import json
from lpdedoc.lpdedoc_codebleu_eval_v1 import ModelCodeGenerator, HybridCodeEvaluator, AdaptiveHybridEvaluator
from lpdedoc.lpdedoc_structcode_eval_v1 import LPStructHybridEvaluator
from lpdedoc.lpdedoc_structcode_eval_v2 import AdaptiveLPStructEvaluator
import os

os.environ["cuda_visible_devices"] = "6"  # GPU 설정 (필요시 조정)

# 공통 출력 포맷 함수
def print_eval_log(case_index, instruction, style_score=None, semantic_score=None, struct_score=None,
                   final_score=None, threshold=0.75):
    result = "PASS" if final_score >= threshold else "FAIL"
    msg = f"[{case_index}] '{instruction[:50]}...'\n"
    if style_score is not None:
        msg += f"  - 스타일 유사도: {style_score:.4f}\n"
    if semantic_score is not None:
        msg += f"  - 의미 유사도  : {semantic_score:.4f}\n"
    if struct_score is not None:
        msg += f"  - 구조 유사도 : {struct_score:.4f}\n"
    msg += f"  -> 최종 점수: {final_score:.4f} / 기준 {threshold} -> {result}\n"
    print(msg)


# v1: LPcodedec + CodeBLEU
def lpbleu_v1(model_path, dataset_path, threshold=0.75, max_samples=None):
    code_generator = ModelCodeGenerator(model_path)
    evaluator = HybridCodeEvaluator(style_weight=0.4, semantic_weight=0.6)

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    for idx, sample in enumerate(test_data, start=1):
        instruction = sample["instruction"]
        reference_code = sample["output"]

        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        eval_result = evaluator.evaluate_best_of_n(reference_code, generated_codes, instruction, lang="python")

        best_score = eval_result["best_result"]["hybrid_score"]
        style_score = eval_result["best_result"]["style_similarity"]
        semantic_score = eval_result["best_result"]["semantic_similarity"]

        print_eval_log(idx, instruction, style_score=style_score,
                       semantic_score=semantic_score,
                       final_score=best_score,
                       threshold=threshold)


# v2: LPcodedec + CodeBLEU (Adaptive)
def lpbleu_v2(model_path, dataset_path, threshold=0.75, max_samples=None):
    code_generator = ModelCodeGenerator(model_path)
    evaluator = AdaptiveHybridEvaluator()

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    for idx, sample in enumerate(test_data, start=1):
        instruction = sample["instruction"]
        reference_code = sample["output"]

        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        eval_result = evaluator.evaluate_with_adaptive_strategy(reference_code, generated_codes,
                                                                instruction, lang="python")

        best_score = eval_result["ensemble_result"]["ensemble_score"]

        print_eval_log(idx, instruction,
                       final_score=best_score,
                       threshold=threshold)


# v3: LPcodedec + StructCoder
def lpstruct_v1(model_path, dataset_path, threshold=0.75, max_samples=None):
    code_generator = ModelCodeGenerator(model_path)
    evaluator = LPStructHybridEvaluator(style_weight=0.5, structural_weight=0.5)

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    for idx, sample in enumerate(test_data, start=1):
        instruction = sample["instruction"]
        reference_code = sample["output"]

        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        eval_result = evaluator.evaluate_best_of_n(reference_code, generated_codes, instruction)

        best_score = eval_result["best_result"]["hybrid_score"]
        style_score = eval_result["best_result"]["style_similarity"]
        struct_score = eval_result["best_result"]["structural_similarity"]

        print_eval_log(idx, instruction,
                       style_score=style_score,
                       struct_score=struct_score,
                       final_score=best_score,
                       threshold=threshold)


# v4: LPcodedec + StructCoder (Adaptive)
def lpstruct_v2(model_path, dataset_path, threshold=0.75, max_samples=None):
    code_generator = ModelCodeGenerator(model_path)
    evaluator = AdaptiveLPStructEvaluator()

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    for idx, sample in enumerate(test_data, start=1):
        instruction = sample["instruction"]
        reference_code = sample["output"]

        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        eval_result = evaluator.evaluate_with_adaptive_strategy(reference_code, generated_codes, instruction)

        best_score = eval_result["strategy_results"][eval_result["optimal_strategy"]]['best_result']["weighted_score"]

        print_eval_log(idx, instruction,
                       final_score=best_score,
                       threshold=threshold)


if __name__ == "__main__":
    MODEL_PATH = "./finetuned_model/finetuned_V2_quantized_pruned"
    DATASET_PATH = "dataset/test_data.json"
    THRESHOLD = 0.75
    MAX_SAMPLES = 15  # 전체 중 몇 개만 평가

    #print("=== LPBLEU V1 ===")
    #lpbleu_v1(MODEL_PATH, DATASET_PATH, threshold=THRESHOLD, max_samples=MAX_SAMPLES)

    #print("\n=== LPBLEU V2 (Adaptive) ===")
    #lpbleu_v2(MODEL_PATH, DATASET_PATH, threshold=THRESHOLD, max_samples=MAX_SAMPLES)

    #print("\n=== LPStruct V1 ===")
    #lpstruct_v1(MODEL_PATH, DATASET_PATH, threshold=THRESHOLD, max_samples=MAX_SAMPLES)

    print("\n=== LPStruct V2 (Adaptive) ===")
    lpstruct_v2(MODEL_PATH, DATASET_PATH, threshold=THRESHOLD, max_samples=MAX_SAMPLES)
