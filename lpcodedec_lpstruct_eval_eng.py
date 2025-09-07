import ast
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os
import time

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

### LPcodedec ###

class LPcodedecAnalyzer:
    def __init__(self):
        self.naming_patterns = {
            'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
            'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'snake_case': re.compile(r'^[a-z_][a-z0-9_]*$'),
            'UPPER_SNAKE_CASE': re.compile(r'^[A-Z_][A-Z0-9_]*$'),
        }
    def extract_lpcodedec_features(self, code: str) -> np.ndarray:
        try:
            tree = ast.parse(code)
        except:
            return np.zeros(10, dtype=np.float32)
        naming = self._analyze_naming_consistency(tree)         # name consistency analysis line 50
        structure = self._analyze_code_structure(code, tree)    # code structure analysis line 82
        readability = self._analyze_readability(code, tree)     # readability analysis line 114
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

    def _analyze_naming_consistency(self, tree: ast.AST) -> Dict[str, float]:   # analyze name consistency based on AST
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
    def _get_consistency(self, names: List[str]) -> float:  # calculate naming consistency
        if not names: return 0.0
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

    def _analyze_code_structure(self, code, tree):      # analyze structure
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
    def _calculate_nesting_depth(self, node):       # calculate code depth
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
    def _analyze_readability(self, code, tree):     # analyze readability
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

### StructCoder ###

class StructCoderAnalyzer:
    def __init__(self):
        self.ast_node_types = [
            'Module', 'FunctionDef', 'ClassDef', 'Return', 'Delete', 'Assign',
            'AugAssign', 'AnnAssign', 'For', 'While', 'If', 'With', 'Raise', 'Try',
            'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal', 'Expr', 'Pass',
            'Break', 'Continue', 'Call', 'Compare', 'BinOp', 'UnaryOp', 'Lambda',
            'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
            'Await', 'Yield', 'YieldFrom'
        ]
    def extract_structural_features(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
        except:
            return self._get_empty_features()
        astf = self._extract_ast_structural_features(tree)      # line 160
        dfgf = self._extract_dataflow_features(tree)            # line 188
        cfgf = self._extract_control_flow_features(tree)        # line 218
        callf = self._extract_call_graph_features(tree)         # line 277
        depf = self._extract_dependency_features(tree)          # line 296
        return {
            'ast_features': astf,
            'dfg_features': dfgf,
            'cfg_features': cfgf,
            'call_graph_features': callf,
            'dependency_features': depf,
            'combined_structural_vector': self._combine_features(astf, dfgf, cfgf, callf, depf)
        }
    def _extract_ast_structural_features(self, tree):   # extract ast features
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
            if not children: leaf_nodes += 1
            else: branching_factors.append(len(children))
            for ch in children: walk(ch, depth+1)
        walk(tree)
        features = []
        for node_type in self.ast_node_types:
            freq = node_counts[node_type] / total_nodes if total_nodes > 0 else 0
            features.append(freq)
        features.extend([
            max_depth, leaf_nodes / total_nodes if total_nodes else 0,
            np.mean(branching_factors) if branching_factors else 0,
            np.std(branching_factors) if branching_factors else 0,
            len(set(node_counts.keys())),
        ])
        return np.array(features, dtype=np.float32)
    def _extract_dataflow_features(self, tree):     
        variables, assignments, usages, def_use_chains = set(), [], [], []
        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.scope_vars = set()
            def visit_Name(self, node):
                variables.add(node.id)
                if isinstance(node.ctx, ast.Store):
                    assignments.append(node.id)
                    self.scope_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    usages.append(node.id)
                    if node.id in self.scope_vars:
                        def_use_chains.append((node.id, 'local'))
                    else:
                        def_use_chains.append((node.id, 'non_local'))
                self.generic_visit(node)
        Visitor().visit(tree)
        num_vars, assignment_counts, usage_counts = len(variables), Counter(assignments), Counter(usages)
        features = [
            num_vars, len(assignments), len(usages),
            len(assignments)/num_vars if num_vars else 0,
            len(usages)/num_vars if num_vars else 0,
            len(assignment_counts)/num_vars if num_vars else 0,
            len(usage_counts)/num_vars if num_vars else 0,
            np.mean(list(assignment_counts.values())) if assignment_counts else 0,
            np.mean(list(usage_counts.values())) if usage_counts else 0,
            len([ch for ch in def_use_chains if ch[1]=='local']) / len(def_use_chains) if def_use_chains else 0
        ]
        return np.array(features, dtype=np.float32)
    def _extract_control_flow_features(self, tree):
        control = {
            'if_count': 0, 'for_count': 0, 'while_count': 0,
            'try_count': 0, 'with_count': 0, 'nested_loops': 0
        }
        function_complexities = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
            def visit_If(self, node):
                control['if_count'] += 1
                self.generic_visit(node)
            def visit_For(self, node):
                control['for_count'] += 1
                self.loop_depth += 1
                if self.loop_depth > 1:
                    control['nested_loops'] += 1
                self.generic_visit(node)
                self.loop_depth -= 1
            def visit_While(self, node):
                control['while_count'] += 1
                self.loop_depth += 1
                if self.loop_depth > 1:
                    control['nested_loops'] += 1
                self.generic_visit(node)
                self.loop_depth -= 1
            def visit_Try(self, node):
                control['try_count'] += 1
                self.generic_visit(node)
            def visit_With(self, node):
                control['with_count'] += 1
                self.generic_visit(node)
            def visit_FunctionDef(self, node):
                c = 1
                for ch in ast.walk(node):
                    if isinstance(ch, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                        c += 1
                function_complexities.append(c)
                self.generic_visit(node)
        Visitor().visit(tree)
        total = sum(control.values())
        features = [
        control['if_count'], control['for_count'], control['while_count'],
        control['try_count'], control['with_count'], control['nested_loops'],
        total,
        np.mean(function_complexities) if function_complexities else 0,
        np.std(function_complexities) if function_complexities else 0,
        max(function_complexities) if function_complexities else 0]
        
        return  np.array(features, dtype=np.float32)

    def _extract_call_graph_features(self, tree):
        calls, defs, builtins = [], [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fn = node.func.id
                    calls.append(fn)
                    if fn in ['len','print','range','int','str','list','dict','set']:
                        builtins.append(fn)
            elif isinstance(node, ast.FunctionDef):
                defs.append(node.name)
        call_counts = Counter(calls)
        features = [
            len(calls), len(defs), len(builtins), len(set(calls)),
            len(builtins)/len(calls) if calls else 0,
            np.mean(list(call_counts.values())) if call_counts else 0,
            max(call_counts.values()) if call_counts else 0
        ]
        return np.array(features, dtype=np.float32)
    def _extract_dependency_features(self, tree):
        dependencies, scopes = [], []
        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.cur = set(); self.stack = []
            def visit_FunctionDef(self, node):
                self.stack.append(self.cur.copy())
                scopes.append(len(self.cur))
                for arg in node.args.args:
                    self.cur.add(arg.arg)
                self.generic_visit(node)
                self.cur = self.stack.pop() if self.stack else set()
            def visit_Assign(self, node):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.cur.add(t.id)
                for var in ast.walk(node.value):
                    if isinstance(var, ast.Name) and isinstance(var.ctx, ast.Load):
                        if var.id in self.cur: dependencies.append(('local', var.id))
                        else: dependencies.append(('global', var.id))
                self.generic_visit(node)
        Visitor().visit(tree)
        local_deps = len([d for d in dependencies if d[0]=='local'])
        global_deps = len([d for d in dependencies if d[0]=='global'])
        features = [
            len(dependencies), local_deps, global_deps,
            local_deps/len(dependencies) if dependencies else 0,
            len(scopes), np.mean(scopes) if scopes else 0
        ]
        return np.array(features, dtype=np.float32)
    def _combine_features(self, astf, dfgf, cfgf, callf, depf):
        return np.concatenate([astf, dfgf, cfgf, callf, depf])
    def _get_empty_features(self) -> Dict[str, Any]:
        return {
            'ast_features': np.zeros(len(self.ast_node_types)+5, dtype=np.float32),
            'dfg_features': np.zeros(10, dtype=np.float32),
            'cfg_features': np.zeros(10, dtype=np.float32),
            'call_graph_features': np.zeros(7, dtype=np.float32),
            'dependency_features': np.zeros(6, dtype=np.float32),
            'combined_structural_vector': np.zeros(len(self.ast_node_types)+38, dtype=np.float32)
        }

### Hybrid Evaluator (Version 1) ###

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

##### Adaptive Evaluator #####

class AdaptiveLPStructEvaluator(HybridLPStructEvaluator):
    """Hybrid evaluator using adaptive weights"""
    def __init__(self):
        super().__init__()
    def evaluate_with_adaptive_strategy(self, reference_code: str, generated_codes: List[str], instruction: str = "") -> Dict[str, Any]:
        code_characteristics = self._analyze_comprehensive_characteristics(reference_code, generated_codes)
        adaptive_weights = self._calculate_adaptive_weights(code_characteristics)
        base_results = []
        for gen_code in generated_codes:
            base_result = self.evaluate_single_pair(reference_code, gen_code)
            # Convert to only style/structure scores
            base_result['style_similarity'] = float(base_result['style_similarity'])
            base_result['structural_similarity'] = float(base_result['structural_similarity'])
            base_results.append(base_result)
        strategy_results = self._evaluate_multiple_strategies(base_results, adaptive_weights)
        optimal_strategy = self._select_optimal_strategy(strategy_results, code_characteristics)
        return {
            'instruction': instruction,
            'reference_code': reference_code,
            'strategy_results': strategy_results,
            'adaptive_weights': adaptive_weights,
            'optimal_strategy': optimal_strategy,
            'code_characteristics': code_characteristics,
            'recommended_result': strategy_results[optimal_strategy]
        }
    def _analyze_comprehensive_characteristics(self, reference_code: str, generated_codes: List[str]) -> Dict[str, Any]:
        ref_lp = self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code)
        gen_lp = [self.lpcodedec_analyzer.extract_lpcodedec_features(code) for code in generated_codes]
        ref_struct = self.structcoder_analyzer.extract_structural_features(reference_code)
        gen_struct = [self.structcoder_analyzer.extract_structural_features(code) for code in generated_codes]
        lp_var = np.std(gen_lp, axis=0) if gen_lp else np.zeros(10)
        comment_var = lp_var[7] if lp_var.shape[0]>=8 else 0.0
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
    def _calculate_cyclomatic_complexity(self, code: str) -> int:       # structure for complexity ratio calculation
        c = 1
        c += code.count('if ')
        c += code.count('elif ')
        c += code.count('for ')
        c += code.count('while ')
        c += code.count('try:')
        c += code.count('except')
        c += code.count('and ')
        c += code.count('or ')
        c += code.count('def ')
        return c
    def _calculate_structural_diversity(self, struct_features_list: List[Dict]) -> float:
        if len(struct_features_list) < 2: return 0.0
        ast_features_array = np.array([
            sf['ast_features'] for sf in struct_features_list if 'ast_features' in sf
        ])
        if ast_features_array.shape[0] == 0: return 0.0
        return np.mean(np.std(ast_features_array, axis=0))
    def _calculate_adaptive_weights(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        style_weight = 0.5
        structural_weight = 0.5
        # Comment Ratio is important
        if characteristics['comment_ratio_variation'] > 0.1:
            style_weight += 0.15
            structural_weight -= 0.15
        # complexity ratio
        if characteristics['complexity_ratio'] > 1.3:
            structural_weight += 0.10
            style_weight -= 0.1
        elif characteristics['complexity_ratio'] < 0.7:
            style_weight -= 0.05
            structural_weight += 0.05
        # structural diversity
        if characteristics["structural_diversity"] > 0.5:
            structural_weight += 0.1
            style_weight -= 0.1
        # high variation features (adjust for style bias)
        if len(characteristics["high_variation_features"]) > 5:
            style_weight += 0.05
            structural_weight -= 0.05
        tot = style_weight + structural_weight
        return {'style': style_weight/tot, 'structural': structural_weight/tot}
    def _evaluate_multiple_strategies(self, base_results: List[Dict], adaptive_weights: Dict[str, float]) -> Dict[str, Dict]:
        strategies = {
            'balanced': {'style': 0.5, 'structural': 0.5},
            'lpcodedec_focused': {'style': 0.7, 'structural': 0.3},
            'structcoder_focused': {'style': 0.3, 'structural': 0.7},
            'adaptive': adaptive_weights
        }
        strategy_results = {}
        for sn, w in strategies.items():
            weighted_results = []
            for r in base_results:
                weighted_score = (w['style']*r['style_similarity'] + w['structural']*r['structural_similarity'])
                wr = r.copy(); wr['weighted_score'] = weighted_score
                weighted_results.append(wr)
            best_result = max(weighted_results, key=lambda x: x['weighted_score'])
            strategy_results[sn] = {'best_result': best_result, 'all_results': weighted_results, 'weights_used': w}
        return strategy_results
    def _select_optimal_strategy(self, strategy_results: Dict, characteristics: Dict[str, Any]) -> str:
        if characteristics['comment_ratio_variation'] > 0.15: return 'lpcodedec_focused'
        if characteristics['structural_diversity'] > 0.6: return 'structcoder_focused'
        if abs(characteristics['complexity_ratio']-1.0) > 0.4: return 'adaptive'
        return 'balanced'

### Code Generator ###

class ModelCodeGenerator:
    def __init__(self, model_path: str, device: str='cuda'):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device); self.model.eval()
    def generate_codes(self, instruction: str, num_candidates: int = 3, max_new_tokens: int = 512) -> List[str]:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            generation_outputs = self.model.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95,
                num_return_sequences=num_candidates, pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        generated_codes = []
        for output in generation_outputs:
            gen_text = self.tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)
            generated_codes.append(self._clean_generated_code(gen_text))
        return generated_codes
    def _clean_generated_code(self, code: str) -> str:
        lines = code.split('\n')
        clean = [line for line in lines if line.strip() and not (line.strip().startswith('###') or line.strip().startswith('```'))]
        return '\n'.join(clean).strip()

### numpy → python dtype conversion function ###

def convert_numpy(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (np.generic,)): return obj.item()
    elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy(i) for i in obj]
    else: return obj

import re

def extract_python_code(text: str) -> str:
    """
    Extract only Python function/code blocks from text.
    If multiple codes are mixed in, retrieves from the first Python function to the end.
    """
    match = re.search(r"(def\s+[a-zA-Z_][a-zA-Z0-9_]*$$.*?$$:[\s\S]+?)(?=^##|<form|<script|$)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"``````")
    if match2:
        return match2.group(1).strip()
    return text.strip()

def main_adaptive_timed():
    print("=== Adaptive LPcodedec+StructCoder Evaluation (include time measurment) ===")
    MODEL = "./finetuned_model/finetuned_V1_quantized_pruned"
    code_generator = ModelCodeGenerator(MODEL)
    evaluator = AdaptiveLPStructEvaluator()

    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_results = []
    test_data = test_data[:15]  # determine number of test data

    total_start_time = time.time()  # inference time measurement
    total_inference_time = 0.0
    total_inference_calls = 0

    for idx, sample in enumerate(test_data):
        instruction = sample["instruction"]
        reference_code = sample["output"]
        print(f"\n[{idx+1}/{len(test_data)}] for adaptive evaluation... {instruction[:80]}...")

        # measure inference time for each generation
        inference_start = time.time()
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        inference_end = time.time()
        inference_duration = inference_end - inference_start

        total_inference_time += inference_duration
        total_inference_calls += len(generated_codes)

        result = evaluator.evaluate_with_adaptive_strategy(reference_code, generated_codes, instruction)
        best_result = result['strategy_results'][result['optimal_strategy']]['best_result']
        if (best_result['hybrid_score'] == 0.0 and 
            best_result['style_similarity'] == 0.0 and 
            best_result['structural_similarity'] == 0.0):
            continue
        print(f"  best strategy: {result['optimal_strategy']}")
        print(f"  best score: {best_result['weighted_score']:.4f}")
        improvement = best_result['weighted_score'] - np.mean(
            [r['weighted_score'] for r in 
            result["strategy_results"][result["optimal_strategy"]]["all_results"]]
        )
        print(f"  score improvement {improvement:.4f}")
        print(f"  style similarity: {best_result['style_similarity']:.4f}")
        print(f"  structural similarity: {best_result['structural_similarity']:.4f}")

        all_results.append({
            "best_hybrid_score": best_result["weighted_score"],
            "score_improvement": improvement,
            "style_similarity": best_result["style_similarity"],
            "structural_similarity": best_result["structural_similarity"]
        })

    # ===== Print overall statistics =====
    print("\n=== Total Evaluation Result ===")
    avg_best = np.mean([r["best_hybrid_score"] for r in all_results])
    avg_improve = np.mean([r["score_improvement"] for r in all_results])
    avg_style = np.mean([r["style_similarity"] for r in all_results])
    avg_struct = np.mean([r["structural_similarity"] for r in all_results])

    total_elapsed_time = time.time() - total_start_time
    avg_inference_time = total_inference_time / total_inference_calls if total_inference_calls else 0

    print(f"Average best hybrid score: {avg_best:.4f}")
    print(f"Average score improvement: {avg_improve:.4f}")
    print(f"Average Style Similarity: {avg_style:.4f}")
    print(f"Average Structural Similarity: {avg_struct:.4f}")
    print(f"Total cost time: {total_elapsed_time:.2f}초")
    print(f"Avaerage inference time per 1 candidates: {avg_inference_time:.2f}초")


def main():
    print("=== LPcodedec + StructCoder hybrid evaluation ===")
    MODEL = "jack0503/code_generate_explain" # any finetuned model to use
    code_generator = ModelCodeGenerator(MODEL)
    evaluator = HybridLPStructEvaluator(style_weight=0.5, structural_weight=0.5)
    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    results = []
    n = min(5, len(test_data))
    print(f"\nevaluate {n} test cases\n")
    for idx, sample in enumerate(test_data[:n]):
        instruction = sample["instruction"]
        reference_code = extract_python_code(sample["output"])
        print(f"[{idx+1}/{n}] 평가 중... Instruction: {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        result = evaluator.evaluate_best_of_n(reference_code, generated_codes, instruction)
        # 간단 출력
        print(f"  ✓ 최고 점수: {case_result['best_hybrid_score']:.4f} (스타일: {case_result['best_style_similarity']:.4f}, 구조: {case_result['best_structural_similarity']:.4f})")
        print(f"  ✓ 점수 향상: {case_result['score_improvement']:.4f}, 다양성: {case_result['score_diversity']:.4f}, 상관관계: {case_result['style_structural_correlation']:.4f}")
        print(f"  ✓ 생성된 코드 예시:\n{case_result['best_generated_code'][:200]}{'...' if len(case_result['best_generated_code']) > 200 else ''}")
    print("\ntest end")

def main_adaptive():
    print("=== Adaptive LPcodedec+StructCoder Evaluation ===")
    #MODEL = "Qwen/Qwen2.5-3B-Instruct"
    MODEL = "jack0503/code_generate_explain"  
    #MODEL = "jack0503/code-usage-model"  
    #MODEL = "./finetuned_model/finetuned_V1_quantized_pruned"  
    code_generator = ModelCodeGenerator(MODEL)
    evaluator = AdaptiveLPStructEvaluator()
    with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    all_results = []

    test_data = test_data[:3]  # determine number of test data
    
    for idx, sample in enumerate(test_data):
        instruction = sample["instruction"]
        reference_code = sample["output"]
        print(f"\n[{idx+1}/{len(test_data)}] for adaptive evaluation... {instruction[:80]}...")
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        result = evaluator.evaluate_with_adaptive_strategy(reference_code, generated_codes, instruction)
        best_result = result['strategy_results'][result['optimal_strategy']]['best_result']

        print(f"  best strategy: {result['optimal_strategy']}")
        print(f"  best score: {best_result['weighted_score']:.4f}")
        improvement = best_result['weighted_score'] - np.mean([r['weighted_score'] for r in result["strategy_results"][result["optimal_strategy"]]["all_results"]])
        print(f"  score improvement: {improvement:.4f}")
        print(f"  style similarity: {best_result['style_similarity']:.4f}")
        print(f"  structural similarity: {best_result['structural_similarity']:.4f}")
    all_results.append({
            "best_hybrid_score": best_result["weighted_score"],
            "score_improvement": improvement,
            "style_similarity": best_result["style_similarity"],
            "structural_similarity": best_result["structural_similarity"]
        })

    # === Print overall statistics ===
    print("\n=== Total Evaluation Result ===")
    avg_best = np.mean([r["best_hybrid_score"] for r in all_results])
    avg_improve = np.mean([r["score_improvement"] for r in all_results])
    avg_style = np.mean([r["style_similarity"] for r in all_results])
    avg_struct = np.mean([r["structural_similarity"] for r in all_results])

    print(f"Average best hybrid score: {avg_best:.4f}")
    print(f"Average score improvement: {avg_improve:.4f}")
    print(f"Average style similarity: {avg_style:.4f}")
    print(f"Average structural similarity: {avg_struct:.4f}")

if __name__ == "__main__":
    main()                  # basic evaluation Best of N
    #main_adaptive()           # Adaptive evaluation
    #main_adaptive_timed()  # Adaptive evaluation with time measurement
