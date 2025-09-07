import ast
import re
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from codebleu import calc_codebleu
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class LPcodedecAnalyzer:
    """LPcodedec 논문의 코딩 스타일 특징 분석기"""
    
    def __init__(self):
        self.naming_patterns = {
            'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
            'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'snake_case': re.compile(r'^[a-z_][a-z0-9_]*$'),
            'UPPER_SNAKE_CASE': re.compile(r'^[A-Z_][A-Z0-9_]*$'),
        }
    
    def extract_lpcodedec_features(self, code: str) -> np.ndarray:
        """LPcodedec 논문의 10가지 스타일 특징 추출"""
        try:
            tree = ast.parse(code)
        except:
            return np.zeros(10, dtype=np.float32)
        
        # 1-4. Naming Consistency 특징들
        naming_features = self._analyze_naming_consistency(tree)
        
        # 5-7. Code Structure 특징들  
        structure_features = self._analyze_code_structure(code, tree)
        
        # 8-10. Readability 특징들
        readability_features = self._analyze_readability(code, tree)
        
        features = [
            naming_features['function_naming'],      # 1. Function Naming Consistency
            naming_features['variable_naming'],      # 2. Variable Naming Consistency
            naming_features['class_naming'],         # 3. Class Naming Consistency
            naming_features['constant_naming'],      # 4. Constant Naming Consistency
            structure_features['indentation_consistency'], # 5. Indentation Consistency
            structure_features['avg_function_length'],     # 6. Function Length
            structure_features['avg_nesting_depth'],       # 7. Nesting Depth
            readability_features['comment_ratio'],         # 8. Comment Ratio
            readability_features['avg_function_name_length'], # 9. Function Name Length
            readability_features['avg_variable_name_length']  # 10. Variable Name Length
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _analyze_naming_consistency(self, tree: ast.AST) -> Dict[str, float]:
        """네이밍 일관성 분석 (LPcodedec Table 8 기준)"""
        functions, variables, classes, constants = [], [], [], []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Name):
                if node.id.isupper():
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
        """가장 일관된 네이밍 패턴의 비율 계산"""
        if not names:
            return 0.0
        
        pattern_counts = {}
        for name in names:
            for pattern_name, pattern in self.naming_patterns.items():
                if pattern.match(name):
                    pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                    break
            else:
                pattern_counts['other'] = pattern_counts.get('other', 0) + 1
        
        return max(pattern_counts.values()) / len(names) if pattern_counts else 0.0
    
    def _analyze_code_structure(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """코드 구조 분석"""
        lines = code.split('\n')
        indentations = []
        function_lengths = []
        nesting_depths = []
        
        # 들여쓰기 분석
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indentations.append(indent)
        
        # 함수 길이 및 중첩 깊이 분석
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 1
                function_lengths.append(func_lines)
                nesting_depths.append(self._calculate_nesting_depth(node))
        
        indent_consistency = 0.0
        if indentations:
            most_common_indent = Counter(indentations).most_common(1)[0][0]
            indent_consistency = indentations.count(most_common_indent) / len(indentations)
        
        return {
            'indentation_consistency': indent_consistency,
            'avg_function_length': np.mean(function_lengths) if function_lengths else 0,
            'avg_nesting_depth': np.mean(nesting_depths) if nesting_depths else 0
        }
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """중첩 깊이 계산"""
        max_depth = 0
        
        def calculate_depth(n, current_depth=0):
            nonlocal max_depth
            if isinstance(n, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(n):
                calculate_depth(child, current_depth)
        
        calculate_depth(node)
        return max_depth
    
    def _analyze_readability(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """가독성 분석"""
        lines = code.split('\n')
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        function_name_lengths = []
        variable_name_lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name_lengths.append(len(node.name))
            elif isinstance(node, ast.Name):
                variable_name_lengths.append(len(node.id))
        
        return {
            'comment_ratio': comment_lines / total_lines if total_lines > 0 else 0,
            'avg_function_name_length': np.mean(function_name_lengths) if function_name_lengths else 0,
            'avg_variable_name_length': np.mean(variable_name_lengths) if variable_name_lengths else 0
        }

# HybridCodeEvaluator -> HybridLPStructEvaluator (기준 코드명으로 통일)
class HybridLPCodeEvaluator:
    """LPcodedec + CodeBLEU 하이브리드 평가기"""
    
    def __init__(self, 
                 style_weight: float = 0.5,
                 structural_weight: float = 0.5):  # semantic_weight -> structural_weight로 통일
        self.lpcodedec_analyzer = LPcodedecAnalyzer()
        self.style_weight = style_weight
        self.structural_weight = structural_weight  # semantic_weight -> structural_weight
        
        # 가중치 정규화
        total_weight = style_weight + structural_weight
        self.style_weight = style_weight / total_weight
        self.structural_weight = structural_weight / total_weight  # semantic_weight -> structural_weight
    
    def evaluate_single_pair(self, reference_code: str, generated_code: str, lang: str = "python") -> Dict[str, Any]:
        """단일 코드 쌍 하이브리드 평가"""
        
        # 1. LPcodedec 스타일 유사도
        ref_style_features = self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code)
        gen_style_features = self.lpcodedec_analyzer.extract_lpcodedec_features(generated_code)
        style_similarity = self._cosine_similarity(ref_style_features, gen_style_features)
        
        # 2. CodeBLEU 의미적 유사도 -> structural_similarity로 통일
        structural_similarity, codebleu_details = self._calculate_codebleu_similarity(
            reference_code, generated_code, lang
        )
        
        # 3. 하이브리드 점수 계산
        hybrid_score = (
            self.style_weight * style_similarity +
            self.structural_weight * structural_similarity  # semantic_weight -> structural_weight
        )
        
        return {
            'hybrid_score': hybrid_score,
            'style_similarity': style_similarity,
            'structural_similarity': structural_similarity,  # semantic_similarity -> structural_similarity
            'style_features_ref': ref_style_features.tolist(),
            'style_features_gen': gen_style_features.tolist(),
            'codebleu_details': codebleu_details,  # structural_details와 같은 역할
            'weights': {
                'style_weight': self.style_weight,
                'structural_weight': self.structural_weight  # semantic_weight -> structural_weight
            }
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _calculate_codebleu_similarity(self, ref_code: str, gen_code: str, lang: str) -> Tuple[float, Dict]:
        """CodeBLEU 의미적 유사도 계산"""
        try:
            result = calc_codebleu([ref_code], [gen_code], lang)
            return result['codebleu'], {
                'ngram_match': result['ngram_match_score'],
                'weighted_ngram_match': result['weighted_ngram_match_score'],
                'syntax_match': result['syntax_match_score'],
                'dataflow_match': result['dataflow_match_score']
            }
        except Exception as e:
            print(f"CodeBLEU 계산 오류: {e}")
            return 0.0, {
                'ngram_match': 0, 'weighted_ngram_match': 0, 
                'syntax_match': 0, 'dataflow_match': 0
            }
    
    def evaluate_best_of_n(self, 
                          reference_code: str,
                          generated_codes: List[str],
                          instruction: str = "",
                          lang: str = "python") -> Dict[str, Any]:
        """Best-of-N 방식 하이브리드 평가"""
        
        if not generated_codes:
            raise ValueError("생성된 코드 목록이 비어있습니다.")
        
        evaluation_results = []
        for i, gen_code in enumerate(generated_codes):
            try:
                result = self.evaluate_single_pair(reference_code, gen_code, lang)
                result['generation_index'] = i
                result['generated_code'] = gen_code
                evaluation_results.append(result)
            except Exception as e:
                print(f"생성 코드 {i} 평가 오류: {e}")
                evaluation_results.append({
                    'hybrid_score': 0.0,
                    'style_similarity': 0.0,
                    'structural_similarity': 0.0,  # semantic_similarity -> structural_similarity
                    'generation_index': i,
                    'generated_code': gen_code,
                    'error': str(e)
                })
        
        # 최고 점수 선택
        best_result = max(evaluation_results, key=lambda x: x['hybrid_score'])
        
        # 통계 계산
        scores = [r['hybrid_score'] for r in evaluation_results]
        style_scores = [r.get('style_similarity', 0) for r in evaluation_results]
        structural_scores = [r.get('structural_similarity', 0) for r in evaluation_results]  # semantic_scores -> structural_scores
        
        return {
            'instruction': instruction,
            'reference_code': reference_code,
            'best_result': best_result,
            'all_results': evaluation_results,
            'n_generations': len(generated_codes),
            'statistics': {
                'hybrid_scores': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                },
                'component_scores': {  # style_scores -> component_scores (기준 코드와 통일)
                    'style': {
                        'mean': np.mean(style_scores),
                        'std': np.std(style_scores),
                        'best_idx': np.argmax(style_scores)
                    },
                    'structural': {  # semantic_scores -> structural (기준 코드와 통일)
                        'mean': np.mean(structural_scores),
                        'std': np.std(structural_scores),
                        'best_idx': np.argmax(structural_scores)
                    }
                }
            },
            'diversity_metrics': {
                'score_diversity': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                'best_improvement': best_result['hybrid_score'] - np.mean(scores),
                'style_structural_correlation': float(np.corrcoef(style_scores, structural_scores)[0,1]) if len(style_scores)>1 else 0  # style_semantic_correlation -> style_structural_correlation
            }
        }

class ModelCodeGenerator:
    """파인튜닝된 모델을 사용한 코드 생성기"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate_codes(self, instruction: str, num_candidates: int = 3, 
                      max_new_tokens: int = 512) -> List[str]:
        """주어진 코드 설명으로 N개의 서로 다른 코드 생성"""
        
        # 프롬프트 구성
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            generation_outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=num_candidates,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 생성된 코드 디코딩 및 정제
        generated_codes = []
        for output in generation_outputs:
            # 입력 부분 제거하고 생성된 부분만 추출
            generated_text = self.tokenizer.decode(
                output[len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            generated_codes.append(generated_text.strip())
        
        return generated_codes

# AdaptiveHybridEvaluator -> AdaptiveLPStructEvaluator (기준 코드명으로 통일)
class AdaptiveLPBLEUEvaluator(HybridLPCodeEvaluator):
    """적응적 가중치를 사용하는 고급 하이브리드 평가기"""
    
    def __init__(self):
        super().__init__()
        
    def evaluate_with_adaptive_strategy(self, 
                                      reference_code: str,
                                      generated_codes: List[str],
                                      instruction: str = "",
                                      lang: str = "python") -> Dict[str, Any]:
        """코드 특성에 따른 적응적 가중치 적용 평가"""
        
        # 기본 평가 수행
        base_results = []
        for gen_code in generated_codes:
            base_result = self.evaluate_single_pair(reference_code, gen_code, lang)
            base_results.append(base_result)
        
        # 코드 특성 분석 -> _analyze_comprehensive_characteristics로 통일
        code_characteristics = self._analyze_comprehensive_characteristics(reference_code, generated_codes)
        
        # 적응적 가중치 계산
        adaptive_weights = self._calculate_adaptive_weights(code_characteristics)
        
        # 다중 전략 평가 -> _evaluate_multiple_strategies로 통일
        strategy_results = {}
        
        # 전략 1: 균등 가중치
        strategy_results['balanced'] = self._evaluate_with_weights(
            base_results, {'style': 0.6, 'structural': 0.4}  # semantic -> structural
        )
        
        # 전략 2: 스타일 중심 (LPcodedec 논문의 주요 발견 반영) -> lpcodedec_focused로 통일
        strategy_results['lpcodedec_focused'] = self._evaluate_with_weights(
            base_results, {'style': 0.7, 'structural': 0.3}  # semantic -> structural
        )
        
        # 전략 3: 의미 중심 (CodeBLEU 강조) -> structcoder_focused로 통일  
        strategy_results['structcoder_focused'] = self._evaluate_with_weights(
            base_results, {'style': 0.3, 'structural': 0.7}  # semantic -> structural
        )
        
        # 전략 4: 적응적 가중치
        strategy_results['adaptive'] = self._evaluate_with_weights(
            base_results, adaptive_weights
        )
        
        # 앙상블 점수 계산 -> _select_optimal_strategy로 통일
        optimal_strategy = self._select_optimal_strategy(strategy_results, code_characteristics)
        
        return {
            'instruction': instruction,
            'reference_code': reference_code,
            'strategy_results': strategy_results,
            'adaptive_weights': adaptive_weights,
            'optimal_strategy': optimal_strategy,  # recommended_strategy -> optimal_strategy
            'code_characteristics': code_characteristics,
            'recommended_result': strategy_results[optimal_strategy]
        }
    
    def _analyze_comprehensive_characteristics(self, reference_code: str, generated_codes: List[str]) -> Dict[str, Any]:
        """코드 특성 분석"""
        ref_features = self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code)
        
        # 코드 복잡도 분석 -> _calculate_cyclomatic_complexity로 통일
        ref_complexity = self._calculate_cyclomatic_complexity(reference_code)
        gen_complexities = [self._calculate_cyclomatic_complexity(code) for code in generated_codes]
        
        # 코드 길이 분석
        ref_length = len(reference_code.split('\n'))
        gen_lengths = [len(code.split('\n')) for code in generated_codes]
        
        # LPcodedec 특징별 변이성 분석
        gen_features = [self.lpcodedec_analyzer.extract_lpcodedec_features(code) for code in generated_codes]
        feature_variations = np.std(gen_features, axis=0) if gen_features else np.zeros(10)
        
        return {
            'comment_ratio_variation': float(feature_variations[7]) if feature_variations.shape[0]>=8 else 0.0,  # 기준 코드와 통일
            'ref_complexity': ref_complexity,
            'avg_gen_complexity': np.mean(gen_complexities),
            'complexity_ratio': np.mean(gen_complexities) / max(ref_complexity, 1),
            'ref_length': ref_length,
            'avg_gen_length': np.mean(gen_lengths),
            'code_length_ratio': np.mean(gen_lengths) / max(ref_length, 1),  # length_ratio -> code_length_ratio (기준 코드와 통일)
            'lpcodedec_feature_variations': feature_variations.tolist(),
            'high_variation_features': np.where(feature_variations > np.mean(feature_variations))[0].tolist(),
            'structural_diversity': 0.0  # 기준 코드와 통일 (여기서는 기본값)
        }
    
    def _calculate_cyclomatic_complexity(self, code: str) -> float:
        """순환 복잡도 기반 코드 복잡도 계산"""
        complexity = 1
        
        # 제어 구조
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('try:')
        complexity += code.count('except')
        complexity += code.count('and ')
        complexity += code.count('or ')
        complexity += code.count('class ')
        complexity += code.count('def ')
        
        return complexity
    
    def _calculate_adaptive_weights(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """코드 특성 기반 적응적 가중치 계산"""
        
        complexity_ratio = characteristics['complexity_ratio']
        code_length_ratio = characteristics['code_length_ratio']  # length_ratio -> code_length_ratio
        high_variation_features = characteristics['high_variation_features']
        
        # 기본 가중치
        style_weight = 0.5
        structural_weight = 0.5  # semantic_weight -> structural_weight
        
        # 복잡도 기반 조정
        if complexity_ratio > 1.3:  # 생성 코드가 더 복잡
            style_weight += 0.1  # 스타일 차이 더 중요
            structural_weight -= 0.1
        elif complexity_ratio < 0.8:  # 생성 코드가 더 단순
            style_weight -= 0.1
            structural_weight += 0.1  # 의미 보존이 더 중요
        
        # 길이 기반 조정
        if code_length_ratio > 1.2:  # 생성 코드가 더 김
            structural_weight += 0.05  # 의미적 정확성 중요
            style_weight -= 0.05
        
        # LPcodedec 특징 변이성 기반 조정
        if len(high_variation_features) > 5:  # 스타일 변이가 큰 경우
            style_weight += 0.1
            structural_weight -= 0.1
        
        # 정규화
        total = style_weight + structural_weight
        return {
            'style': style_weight / total,
            'structural': structural_weight / total  # semantic -> structural
        }
    
    def _evaluate_multiple_strategies(self, base_results: List[Dict], weights: Dict[str, float]) -> Dict[str, Any]:
        """주어진 가중치로 재평가"""
        weighted_results = []
        
        for result in base_results:
            weighted_score = (
                weights['style'] * result['style_similarity'] +
                weights['structural'] * result['structural_similarity']  # semantic -> structural
            )
            
            weighted_result = result.copy()
            weighted_result['weighted_score'] = weighted_score
            weighted_results.append(weighted_result)
        
        best_result = max(weighted_results, key=lambda x: x['weighted_score'])
        
        return {
            'best_result': best_result,
            'all_results': weighted_results,
            'weights_used': weights
        }
    
    def _evaluate_with_weights(self, base_results: List[Dict], weights: Dict[str, float]) -> Dict[str, Any]:
        """주어진 가중치로 재평가 (기준 코드 메서드명 사용)"""
        return self._evaluate_multiple_strategies(base_results, weights)
    
    def _select_optimal_strategy(self, strategy_results: Dict, characteristics: Dict) -> str:
        """코드 특성 기반 최적 전략 권장"""
        
        # LPcodedec 논문의 주요 발견 반영
        complexity_ratio = characteristics['complexity_ratio']
        high_variation_features = characteristics['high_variation_features']
        
        # Comment Ratio가 논문에서 가장 중요한 특징으로 나타남
        if 7 in high_variation_features:  # Comment Ratio는 인덱스 7
            return 'lpcodedec_focused'  # style_focused -> lpcodedec_focused
        
        # 복잡도 차이가 클 때
        if abs(complexity_ratio - 1.0) > 0.3:
            return 'adaptive'
        
        # 기본적으로는 균형잡힌 전략
        return 'balanced'

# 사용 예시
def main():  # main_v1 -> main (기준 코드명으로 통일)
    # 1. 모델 및 평가기 초기화
    MODEL_PATH = "jack0503/code_generate_explain"  # 파인튜닝된 모델 경로
    code_generator = ModelCodeGenerator(MODEL_PATH)
    evaluator = HybridLPCodeEvaluator(style_weight=0.4, structural_weight=0.6)  # semantic_weight -> structural_weight
    
    # 2. 테스트 데이터셋 로드
    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    all_case_results = []
    
    print(f"총 {len(test_data)}개 테스트 케이스 평가 시작...\n")
    
    for idx, sample in enumerate(test_data[:5]):  # 처음 5개만 테스트
        instruction = sample["instruction"]
        reference_code = sample["output"]
        
        print(f"[{idx+1}/{5}] 평가 중...")
        print(f"Instruction: {instruction[:100]}...")
        
        # 3. 코드 설명으로 3개 코드 생성
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        
        # 4. Best-of-3 평가 수행
        result = evaluator.evaluate_best_of_n(
            reference_code,
            generated_codes,
            instruction,
            lang="python"
        )
        
        # 5. 결과 저장
        case_result = {
            "case_index": idx,
            "instruction": instruction,
            "reference_code": reference_code,
            "best_candidate_index": result["best_result"]["generation_index"],
            "best_hybrid_score": result["best_result"]["hybrid_score"],
            "best_style_similarity": result["best_result"]["style_similarity"],
            "best_structural_similarity": result["best_result"]["structural_similarity"],  # best_semantic_similarity -> best_structural_similarity
            "score_improvement": result["diversity_metrics"]["best_improvement"],
            "score_diversity": result["diversity_metrics"]["score_diversity"],
            "all_scores": [
                {
                    "index": r["generation_index"],
                    "hybrid_score": r["hybrid_score"],
                    "style_similarity": r.get("style_similarity", 0),
                    "structural_similarity": r.get("structural_similarity", 0)  # semantic_similarity -> structural_similarity
                }
                for r in result["all_results"] if "error" not in r
            ]
        }
        all_case_results.append(case_result)
        
        # 개별 결과 출력
        print(f"  최고 점수: {result['best_result']['hybrid_score']:.4f}")
        print(f"  선택된 후보: {result['best_result']['generation_index']}번")
        print(f"  점수 개선: {result['diversity_metrics']['best_improvement']:.4f}")
        print(f"  스타일 유사도: {result['best_result']['style_similarity']:.4f}")
        print(f"  구조 유사도: {result['best_result']['structural_similarity']:.4f}\n")  # 의미적 유사도 -> 구조 유사도
    
    # 6. 전체 결과 저장
    with open("./lpdedoc_result_json/hybrid_lpbleu_results.json", "w", encoding="utf-8") as f:
        json.dump(all_case_results, f, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else int(o) if isinstance(o, (np.int32, np.int64)) else o)
    
    # 7. 전체 통계 출력
    print("=== 전체 평가 결과 ===")
    avg_best_score = np.mean([r["best_hybrid_score"] for r in all_case_results])
    avg_improvement = np.mean([r["score_improvement"] for r in all_case_results])
    avg_style = np.mean([r["best_style_similarity"] for r in all_case_results])
    avg_structural = np.mean([r["best_structural_similarity"] for r in all_case_results])  # avg_semantic -> avg_structural
    
    print(f"평균 최고 하이브리드 점수: {avg_best_score:.4f}")
    print(f"평균 점수 개선도: {avg_improvement:.4f}")
    print(f"평균 스타일 유사도: {avg_style:.4f}")
    print(f"평균 구조 유사도: {avg_structural:.4f}")  # 의미적 유사도 -> 구조 유사도

def main_adaptive():  # main_v2 -> main_adaptive (기준 코드명으로 통일)
    MODEL_PATH = "jack0503/code_generate_explain"  # 파인튜닝된 모델 경로
    code_generator = ModelCodeGenerator(MODEL_PATH)
    evaluator = AdaptiveLPBLEUEvaluator()
    
    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    all_results = []
    
    for idx, sample in enumerate(test_data[:3]):  # 3개 샘플 테스트
        instruction = sample["instruction"]
        reference_code = sample["output"]
        
        print(f"\n[{idx+1}/3] 적응적 평가 중...")
        print(f"Instruction: {instruction[:80]}...")
        
        # 코드 생성
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        
        # 적응적 평가 수행
        result = evaluator.evaluate_with_adaptive_strategy(
            reference_code, generated_codes, instruction, lang="python"
        )
        
        # 결과 출력
        print(f"최적 전략: {result['optimal_strategy']}")  # 권장 전략 -> 최적 전략
        print(f"적응적 가중치: 스타일={result['adaptive_weights']['style']:.3f}, "
              f"구조={result['adaptive_weights']['structural']:.3f}")  # 의미 -> 구조
        
        # 전략별 최고 성능 비교
        print("전략별 최고 점수:")
        for strategy, strategy_result in result['strategy_results'].items():
            best_score = strategy_result['best_result']['weighted_score']
            print(f"  {strategy}: {best_score:.4f}")
        
        all_results.append(result)
    
    # 결과 저장
    with open("./lpdedoc_result_json/adaptive_lpbleu_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else int(o) if isinstance(o, (np.int32, np.int64)) else o)

if __name__ == "__main__":
    main()  # main_v1 -> main
    #main_adaptive()  # main_v2 -> main_adaptive