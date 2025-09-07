import ast
import re
import json
import numpy as np
from typing import Dict, List, Any
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os
import time

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


### LPcodedec Analyzer ###
class LPcodedecAnalyzer:
    def __init__(self):
        # 함수, 변수, 클래스, 상수등의 이름 규칙 패턴 정의 (카멜케이스, 파스칼케이스, 스네이크케이스, 대문자 스네이크)
        self.naming_patterns = {
            'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
            'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'snake_case': re.compile(r'^[a-z_][a-z0-9_]*$'),
            'UPPER_SNAKE_CASE': re.compile(r'^[A-Z_][A-Z0-9_]*$'),
        }

    def extract_lpcodedec_features(self, code: str) -> np.ndarray:
        """
        코드에서 스타일 관련 10차원 특징 벡터를 추출함.
        - 이름 규칙 일관성(함수, 변수, 클래스, 상수)
        - 들여쓰기 일관성, 함수 길이, 중첩 깊이
        - 주석 비율, 함수/변수 이름 길이 평균
        """
        try:
            tree = ast.parse(code)
        except:
            return np.zeros(10, dtype=np.float32)

        naming = self._analyze_naming_consistency(tree)
        structure = self._analyze_code_structure(code, tree)
        readability = self._analyze_readability(code, tree)

        features = [
            naming['function_naming'],
            naming['variable_naming'],
            naming['class_naming'],
            naming['constant_naming'],
            structure['indentation_consistency'],
            structure['avg_function_length'],
            structure['avg_nesting_depth'],
            readability['comment_ratio'],
            readability['avg_function_name_length'],
            readability['avg_variable_name_length']
        ]
        return np.array(features, dtype=np.float32)

    def _analyze_naming_consistency(self, tree: ast.AST) -> Dict[str, float]:
        functions, variables, classes, constants = [], [], [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Name):
                if node.id.isupper() and len(node.id) > 1:
                    constants.append(node.id)
                else:
                    variables.append(node.id)
        return {
            'function_naming': self._get_consistency(functions),
            'variable_naming': self._get_consistency(variables),
            'class_naming': self._get_consistency(classes),
            'constant_naming': self._get_consistency(constants)
        }

    def _get_consistency(self, names: List[str]) -> float:
        """
        이름 규칙 일관성을 0~1로 점수화 
        가장 많이 등장하는 패턴 비율을 반환
        """
        if not names:
            return 0.0
        p_count = {}
        for name in names:
            matched = False
            for pname, ptn in self.naming_patterns.items():
                if ptn.match(name):
                    p_count[pname] = p_count.get(pname, 0) + 1
                    matched = True
                    break
            if not matched:
                p_count['other'] = p_count.get('other', 0) + 1
        return max(p_count.values()) / len(names) if p_count else 0.0

    def _analyze_code_structure(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """
        들여쓰기 일관성, 함수 길이 평균, 함수 내 중첩 깊이 평균 계산
        """
        lines = code.split('\n')
        indent = []
        funlens = []
        nests = []

        for line in lines:
            if line.strip():
                ind = len(line) - len(line.lstrip())
                if ind > 0:
                    indent.append(ind)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') and node.end_lineno else 1
                funlens.append(func_lines)
                nests.append(self._calculate_nesting_depth(node))

        ind_consist = (Counter(indent).most_common(1)[0][1] / len(indent)) if indent else 0.0

        return {
            'indentation_consistency': ind_consist,
            'avg_function_length': np.mean(funlens) if funlens else 0,
            'avg_nesting_depth': np.mean(nests) if nests else 0
        }

    def _calculate_nesting_depth(self, node):
        """재귀적으로 중첩 깊이 계산 (for, while, if, with, try, except 노드)"""
        max_depth = 0
        def walk(n, cur=0):
            nonlocal max_depth
            if isinstance(n, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.ExceptHandler)):
                cur += 1
                max_depth = max(max_depth, cur)
            for child in ast.iter_child_nodes(n):
                walk(child, cur)
        walk(node)
        return max_depth

    def _analyze_readability(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """전체 코드에서 주석 비율과 함수명, 변수명 길이 평균 계산"""
        lines = code.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        comment_lines = sum(1 for line in lines if line.strip().startswith('#') or '"""' in line or "'''" in line)

        fnlen, varlen = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fnlen.append(len(node.name))
            elif isinstance(node, ast.Name):
                varlen.append(len(node.id))

        return {
            'comment_ratio': comment_lines / total_lines if total_lines else 0,
            'avg_function_name_length': np.mean(fnlen) if fnlen else 0,
            'avg_variable_name_length': np.mean(varlen) if varlen else 0
        }


### StructCoder Analyzer ###
class StructCoderAnalyzer:
    def __init__(self):
        # AST 노드 타입 목록 (이름 그대로 AST 각종 노드 타입)
        self.ast_node_types = [
            'Module', 'FunctionDef', 'ClassDef', 'Return', 'Delete', 'Assign',
            'AugAssign', 'AnnAssign', 'For', 'While', 'If', 'With', 'Raise', 'Try',
            'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal', 'Expr', 'Pass',
            'Break', 'Continue', 'Call', 'Compare', 'BinOp', 'UnaryOp', 'Lambda',
            'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
            'Await', 'Yield', 'YieldFrom'
        ]

    def extract_structural_features(self, code: str) -> Dict[str, Any]:
        """
        AST를 파싱하여 구조적 특징들 추출.  
        - AST 노드 타입별 빈도, 트리 최대 깊이, 잎 노드 비율, 브랜칭 팩터 평균/표준편차, 사용 노드 타입 수
        - 데이터 흐름, 제어 흐름, 함수 호출 그래프, 의존성 특징도 함께 추출
        """
        try:
            tree = ast.parse(code)
        except:
            return self._get_empty_features()

        astf = self._extract_ast_structural_features(tree)
        dfgf = self._extract_dataflow_features(tree)
        cfgf = self._extract_control_flow_features(tree)
        callf = self._extract_call_graph_features(tree)
        depf = self._extract_dependency_features(tree)

        return {
            'ast_features': astf,
            'dfg_features': dfgf,
            'cfg_features': cfgf,
            'call_graph_features': callf,
            'dependency_features': depf,
            'combined_structural_vector': self._combine_features(astf, dfgf, cfgf, callf, depf)
        }

    def _extract_ast_structural_features(self, tree):
        """
        AST 트리에서
        - 노드 타입 등장비율, 최대 깊이, 잎 노드 비율, 브랜칭 팩터 평균 및 표준편차,
        - 사용된 노드 타입 종류 개수를 추출하는 내부 함수
        """
        node_counts = defaultdict(int)
        total_nodes = 0
        max_depth = 0
        leaf_nodes = 0
        branching_factors = []

        def walk(node, depth=0):
            nonlocal max_depth, total_nodes, leaf_nodes
            node_type = type(node).__name__
            node_counts[node_type] += 1
            total_nodes += 1
            max_depth = max(max_depth, depth)
            children = list(ast.iter_child_nodes(node))
            if not children:
                leaf_nodes += 1
            else:
                branching_factors.append(len(children))
            for ch in children:
                walk(ch, depth + 1)

        walk(tree)

        features = []
        for node_type in self.ast_node_types:
            freq = node_counts[node_type] / total_nodes if total_nodes > 0 else 0
            features.append(freq)

        features.extend([
            max_depth,
            leaf_nodes / total_nodes if total_nodes else 0,
            np.mean(branching_factors) if branching_factors else 0,
            np.std(branching_factors) if branching_factors else 0,
            len(set(node_counts.keys())),
        ])

        return np.array(features, dtype=np.float32)


### AdaptiveLPStructEvaluator 예시 (가중치 조정 관련 부분 발췌) ###
class HybridLPStructEvaluator:
    def __init__(self, style_weight=0.5, structural_weight=0.5):
        self.lpcodedec_analyzer = LPcodedecAnalyzer()
        self.structcoder_analyzer = StructCoderAnalyzer()
        tw = style_weight + structural_weight
        self.style_weight = style_weight / tw
        self.structural_weight = structural_weight / tw

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0: return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
    def _calculate_structural_similarity(self, ref, gen):
        ref_s = self.structcoder_analyzer.extract_structural_features(ref)
        gen_s = self.structcoder_analyzer.extract_structural_features(gen)
        sim = {}
        def safe_sim(vname):
            try:
                return self._cosine_similarity(ref_s[vname], gen_s[vname])
            except: return 0.0
        sim["ast_similarity"]  = safe_sim("ast_features")
        sim["dfg_similarity"]  = safe_sim("dfg_features")
        sim["cfg_similarity"]  = safe_sim("cfg_features")
        sim["call_similarity"] = safe_sim("call_graph_features")
        sim["dependency_similarity"] = safe_sim("dependency_features")
        struct_score = (0.4*sim["ast_similarity"] + 0.3*sim["dfg_similarity"]
                        + 0.2*sim["cfg_similarity"] + 0.07*sim["call_similarity"]
                        + 0.03*sim["dependency_similarity"])
        return float(struct_score), sim
    def evaluate_single_pair(self, reference_code, generated_code) -> Dict[str,Any]:
        style_score = self._cosine_similarity(
            self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code),
            self.lpcodedec_analyzer.extract_lpcodedec_features(generated_code)
        )
        struct_score, struct_details = self._calculate_structural_similarity(reference_code, generated_code)
        hybrid_score = self.style_weight * style_score + self.structural_weight * struct_score
        return {
            "hybrid_score": hybrid_score,
            "style_similarity": style_score,
            "structural_similarity": struct_score,
            "structural_details": struct_details
        }
    def evaluate_best_of_n(self, reference_code: str, candidates: List[str], instruction: str="") -> Dict[str,Any]:
        results = []
        for idx, candidate in enumerate(candidates):
            try:
                r = self.evaluate_single_pair(reference_code, candidate)
                r["generation_index"] = idx
                r["generated_code"] = candidate
                results.append(r)
            except Exception as e:
                results.append({
                    "generation_index": idx,
                    "generated_code": candidate,
                    "error": str(e),
                    "hybrid_score": 0.0,
                    "style_similarity": 0.0,
                    "structural_similarity": 0.0
                })
        best = max(results, key=lambda x: x["hybrid_score"])
        valid = [r for r in results if "error" not in r]
        scores = [r['hybrid_score'] for r in valid] if valid else [0.0]
        style_scores = [r['style_similarity'] for r in valid] if valid else [0.0]
        struct_scores = [r['structural_similarity'] for r in valid] if valid else [0.0]
        return {
            'instruction': instruction,
            'reference_code': reference_code,
            'best_result': best,
            'all_results': results,
            'n_generations': len(candidates),
            'statistics': {
                'hybrid_scores': {'mean': np.mean(scores),'std': np.std(scores),'min': np.min(scores),'max': np.max(scores)},
                'component_scores': {
                    'style': {'mean': np.mean(style_scores),'best_idx': int(np.argmax(style_scores))},
                    'structural': {'mean': np.mean(struct_scores),'best_idx': int(np.argmax(struct_scores))}
                }
            },
            'diversity_metrics': {
                'score_diversity': np.std(scores)/np.mean(scores) if np.mean(scores)>0 else 0,
                'best_improvement': best['hybrid_score'] - np.mean(scores) if scores else 0,
                'style_structural_correlation': float(np.corrcoef(style_scores, struct_scores)[0,1]) if len(style_scores)>1 else 0
            }
        }

class AdaptiveLPStructEvaluator(HybridLPStructEvaluator):
    def __init__(self):
        super().__init__()

    def _analyze_comprehensive_characteristics(self, reference_code: str, generated_codes: List[str]) -> Dict[str, Any]:
        """
        평가할 참조 코드와 생성된 코드들에 대해 다음 특징들을 분석:
        - LPcodedec style feature 변화량 (comment ratio 포함)
        - 참조 및 생성 코드 복잡도 (cyclomatic complexity)
        - 생성된 코드들의 구조적 다양성
        """
        ref_lp = self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code)
        gen_lp = [self.lpcodedec_analyzer.extract_lpcodedec_features(code) for code in generated_codes]
        ref_struct = self.structcoder_analyzer.extract_structural_features(reference_code)
        gen_struct = [self.structcoder_analyzer.extract_structural_features(code) for code in generated_codes]

        lp_var = np.std(gen_lp, axis=0) if gen_lp else np.zeros(10)
        comment_var = lp_var[7] if lp_var.shape >= 8 else 0.0

        ref_complexity = self._calculate_cyclomatic_complexity(reference_code)
        gen_complexities = [self._calculate_cyclomatic_complexity(c) for c in generated_codes]

        struct_diversity = self._calculate_structural_diversity(gen_struct)

        return {
            'comment_ratio_variation': float(comment_var),
            'ref_complexity': ref_complexity,
            'avg_gen_complexity': float(np.mean(gen_complexities)),
            'complexity_ratio': float(np.mean(gen_complexities)) / max(ref_complexity, 1),
            'lpcodedec_feature_variations': lp_var.tolist(),
            'high_variation_features': list(np.where(lp_var > np.mean(lp_var))[0]),
            'structural_diversity': float(struct_diversity),
            'code_length_ratio': float(np.mean([len(c.split('\n')) for c in generated_codes]) / len(reference_code.split('\n')))
        }

def main():
    print("=== LPcodedec + StructCoder Hybrid Evaluation System Start ===")
    MODEL = "jack0503/code_generate_explain"  # 또는 "Qwen/Qwen2.5-3B-Instruct"
    code_generator = ModelCodeGenerator(MODEL)
    evaluator = HybridLPStructEvaluator(style_weight=0.5, structural_weight=0.5)

    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    n = min(5, len(test_data))
    print(f"\nStarting evaluation for {n} test cases...\n")
    for idx, sample in enumerate(test_data[:n]):
        instruction = sample["instruction"]
        reference_code = extract_python_code(sample["output"])
        print(f"[{idx + 1}/{n}] Evaluating... Instruction: {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        result = evaluator.evaluate_best_of_n(reference_code, generated_codes, instruction)

        case_result = {
            "case_index": idx,
            "instruction": instruction,
            "reference_code": reference_code,
            "best_hybrid_score": result["best_result"]["hybrid_score"],
            "best_style_similarity": result["best_result"]["style_similarity"],
            "best_structural_similarity": result["best_result"]["structural_similarity"],
            "score_improvement": result["diversity_metrics"]["best_improvement"],
            "score_diversity": result["diversity_metrics"]["score_diversity"],
            "style_structural_correlation": result["diversity_metrics"]["style_structural_correlation"],
            "component_best_indices": result["statistics"]["component_scores"],
            "structural_details": result["best_result"]["structural_details"]
        }
        results.append(case_result)
        print(f"  ✓ Best score: {case_result['best_hybrid_score']:.4f} (Style: {case_result['best_style_similarity']:.4f}, Structure: {case_result['best_structural_similarity']:.4f})")

    with open("lpstruct_pure_hybrid_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy(results), f, ensure_ascii=False, indent=2)
    print("\n== Results saved successfully ==")


def main_adaptive():
    print("=== Adaptive LPcodedec+StructCoder Evaluation Start ===")
    MODEL = "./finetuned_model/finetuned_V1_quantized_pruned"  # 또는 "Qwen/Qwen2.5-3B-Instruct"
    code_generator = ModelCodeGenerator(MODEL)
    evaluator = AdaptiveLPStructEvaluator()

    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_results = []
    test_data = test_data[:15]  # Number of test datasets to use

    for idx, sample in enumerate(test_data):
        instruction = sample["instruction"]
        reference_code = sample["output"]
        print(f"\n[{idx + 1}/{len(test_data)}] Performing adaptive evaluation... {instruction[:80]}...")
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        result = evaluator.evaluate_with_adaptive_strategy(reference_code, generated_codes, instruction)
        best_result = result['strategy_results'][result['optimal_strategy']]['best_result']

        print(f"  Optimal strategy: {result['optimal_strategy']}")
        print(f"  Best score: {best_result['weighted_score']:.4f}")
        improvement = best_result['weighted_score'] - np.mean([r['weighted_score'] for r in result["strategy_results"][result["optimal_strategy"]]["all_results"]])
        print(f"  Score improvement: {improvement:.4f}")
        print(f"  Style similarity: {best_result['style_similarity']:.4f}")
        print(f"  Structural similarity: {best_result['structural_similarity']:.4f}")

        all_results.append({
            "best_hybrid_score": best_result["weighted_score"],
            "score_improvement": improvement,
            "style_similarity": best_result["style_similarity"],
            "structural_similarity": best_result["structural_similarity"]
        })

    print("\n=== Overall Evaluation Results ===")
    avg_best = np.mean([r["best_hybrid_score"] for r in all_results])
    avg_improve = np.mean([r["score_improvement"] for r in all_results])
    avg_style = np.mean([r["style_similarity"] for r in all_results])
    avg_struct = np.mean([r["structural_similarity"] for r in all_results])

    print(f"Average best hybrid score: {avg_best:.4f}")
    print(f"Average score improvement: {avg_improve:.4f}")
    print(f"Average style similarity: {avg_style:.4f}")
    print(f"Average structural similarity: {avg_struct:.4f}")

    with open("./lpdedoc_result_json/adaptive_lpstruct_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy(all_results), f, ensure_ascii=False, indent=2)
    print("\n== Results saved successfully ==")

if __name__ == "__main__":
    main_adaptive()  # Use main() for standard evaluation, main_adaptive() for adaptive strategy evaluation
    # main()  # Uncomment to run standard evaluation