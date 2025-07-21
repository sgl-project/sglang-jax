import threading
import time
import json
import os
from functools import wraps
from typing import Dict, List, Union, Any, Optional, Callable

import torch
import jax.numpy as jnp


class TensorJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):  # PyTorch tensor
            try:
                return {
                    '__tensor_type__': 'pytorch',
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'data': obj.cpu().numpy().tolist() if obj.numel() < 100 else f'<tensor too large: {obj.shape}>'
                }
            except Exception:
                return {
                    '__tensor_type__': 'pytorch',
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'data': f'<cannot serialize tensor: {obj.shape}>'
                }
        elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):  # JAX array or similar
            try:
                return {
                    '__tensor_type__': 'jax',
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'data': obj.tolist() if obj.size < 100 else f'<array too large: {obj.shape}>'
                }
            except Exception:
                return {
                    '__tensor_type__': 'jax',
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'data': f'<cannot serialize array: {obj.shape}>'
                }
        try:
            return str(obj)
        except Exception:
            return f'<non-serializable object: {type(obj).__name__}>'


class UnifiedDebugTracer:    
    def __init__(self):
        self.enabled = True
        self.records = {}
        self.lock = threading.Lock()
        
        self._inference_session_active = False
        self._accumulated_inputs = []
        self._accumulated_outputs = []
        self._session_start_time = None
        self._forward_count = 0
        self._current_step = 0
        
        self._auto_save_on_destroy = True
        self._max_steps_per_session = 200
        self._session_timeout = 300
        
        self._tokenizer = None
        self._model_class_name = None
    
    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer
    
    def set_model(self, model_obj, tokenizer=None):
        if hasattr(model_obj, '__class__'):
            class_name = model_obj.__class__.__name__
            self._model_class_name = class_name
            print(f"Model class automatically set to: {class_name}")
        
        if tokenizer is not None:
            self._tokenizer = tokenizer
            print(f"Tokenizer set from model setup")
        
        elif hasattr(model_obj, 'config'):
            try:
                from sglang.srt.hf_transformers_utils import get_tokenizer
                model_path = getattr(model_obj.config, '_name_or_path', None)
                if model_path:
                    auto_tokenizer = get_tokenizer(model_path, trust_remote_code=True)
                    self._tokenizer = auto_tokenizer
                    print(f"Tokenizer automatically extracted from model config: {model_path}")
            except Exception as e:
                print(f"Warning: Could not auto-extract tokenizer from model: {str(e)}")
        
        return self._model_class_name
    
    def set_auto_save_config(self, auto_save_on_destroy=True, max_steps_per_session=200, session_timeout=300):
        self._auto_save_on_destroy = auto_save_on_destroy
        self._max_steps_per_session = max_steps_per_session
        self._session_timeout = session_timeout
        print(f"Auto-save config updated: Destroy={auto_save_on_destroy}, MaxSteps={max_steps_per_session}, Timeout={session_timeout}s")
    
    def start_session(self):
        if self._inference_session_active:
            print("Inference session already active, ending current session first...")
            self.end_session()
        
        self._inference_session_active = True
        self._accumulated_inputs = []
        self._accumulated_outputs = []
        self._forward_count = 0
        self._current_step = 0 
        self._session_start_time = time.time()
        self.clear_records()
        print("Inference session started, beginning to accumulate records...")
    
    def end_session(self, save_json=True):
        if not self._inference_session_active:
            print("No active inference session")
            return None
        
        self._inference_session_active = False
        
        if save_json:
            filepath = self._save_complete_session()
            print(f"Inference session ended, complete record saved to: {filepath}")
            return filepath
        else:
            print("Inference session ended, JSON not saved")
            return None
    
    def is_session_active(self):
        return self._inference_session_active
    
    def should_auto_save(self):
        if not self._inference_session_active:
            return False
        
        # Check step limit
        if self._forward_count >= self._max_steps_per_session:
            print(f"Reached maximum steps limit ({self._max_steps_per_session}), auto-saving session...")
            return True
        
        # Check timeout
        if self._session_start_time and (time.time() - self._session_start_time) > self._session_timeout:
            print(f"Session timeout ({self._session_timeout}s), auto-saving session...")
            return True
        
        return False
    
    def accumulate_step(self, input_data: Dict, output_data: Dict):
        if not self._inference_session_active:
            return
        
        # Increment step counter for this forward pass
        self._current_step += 1
        
        step_input = {
            "step": self._current_step,
            "forward_count": self._forward_count,
            **input_data
        }
        
        step_output = {
            "step": self._current_step,
            "forward_count": self._forward_count,
            **output_data
        }
        
        # Decode text if tokenizer available
        if self._tokenizer and "input_ids" in input_data:
            step_input.update(self._decode_input_text(input_data["input_ids"]))
        
        if self._tokenizer and "logits" in output_data:
            step_output.update(self._decode_output_text(output_data["logits"]))
        
        self._accumulated_inputs.append(step_input)
        self._accumulated_outputs.append(step_output)
        self._forward_count += 1
    
    def _decode_input_text(self, input_ids):
        try:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.cpu().tolist()
            
            if isinstance(input_ids[0], list):  # Batch
                input_texts = []
                for seq in input_ids:
                    input_texts.append(self._tokenizer.decode(seq, skip_special_tokens=False))
                return {
                    "input_text": input_texts,
                    "input_text_clean": [self._tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]
                }
            else:
                return {
                    "input_text": self._tokenizer.decode(input_ids, skip_special_tokens=False),
                    "input_text_clean": self._tokenizer.decode(input_ids, skip_special_tokens=True)
                }
        except Exception as e:
            return {"input_decode_error": str(e)}
    
    def _decode_output_text(self, logits):
        try:
            if isinstance(logits, torch.Tensor):
                predicted_token_ids = torch.argmax(logits, dim=-1).cpu().tolist()
                if isinstance(predicted_token_ids, int):
                    predicted_token_ids = [predicted_token_ids]
                
                predicted_tokens = []
                for token_id in predicted_token_ids:
                    token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
                    predicted_tokens.append({
                        "token_id": token_id,
                        "token_text": token_text,
                        "token_text_clean": self._tokenizer.decode([token_id], skip_special_tokens=True)
                    })
                
                top_k_data = {}
                if logits.shape[-1] > 1:
                    top_k = min(5, logits.shape[-1])
                    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
                    top_predictions = []
                    
                    for logit_vals, token_ids in zip(top_logits.cpu(), top_indices.cpu()):
                        if len(logit_vals.shape) == 0:
                            logit_vals = [logit_vals.item()]
                            token_ids = [token_ids.item()]
                        else:
                            logit_vals = logit_vals.tolist()
                            token_ids = token_ids.tolist()
                        
                        seq_predictions = []
                        for logit_val, token_id in zip(logit_vals, token_ids):
                            token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
                            seq_predictions.append({
                                "token_id": token_id,
                                "logit": logit_val,
                                "token_text": token_text,
                                "probability": float(torch.softmax(torch.tensor([logit_val]), dim=0)[0].item())
                            })
                        top_predictions.append(seq_predictions)
                    
                    top_k_data["top_k_predictions"] = top_predictions
                
                return {
                    "predicted_tokens": predicted_tokens,
                    **top_k_data
                }
                
        except Exception as e:
            return {"output_decode_error": str(e)}
        
        return {}
    
    def _save_complete_session(self):
        try:
            all_records = self.get_records()
            
            complete_conversation = ""
            all_tokens = []
            
            if self._tokenizer and self._accumulated_inputs and self._accumulated_outputs:
                try:
                    for step_input in self._accumulated_inputs:
                        if "input_ids" in step_input:
                            input_ids = step_input["input_ids"]
                            if isinstance(input_ids, list):
                                if len(input_ids) > 0 and isinstance(input_ids[0], list):
                                    all_tokens.extend(input_ids[0])
                                else:
                                    all_tokens.extend(input_ids)
                    
                    for step_output in self._accumulated_outputs:
                        predicted_tokens = step_output.get("predicted_tokens", [])
                        if isinstance(predicted_tokens, list) and len(predicted_tokens) > 0:
                            if isinstance(predicted_tokens[0], dict):
                                all_tokens.extend([t.get("token_id", 0) for t in predicted_tokens])
                    
                    if all_tokens:
                        complete_conversation = self._tokenizer.decode(all_tokens, skip_special_tokens=True)
                except Exception as e:
                    print(f"Error decoding complete conversation: {str(e)}")
                    complete_conversation = "Decoding failed"
            
            debug_info = {
                "session_info": {
                    "total_forward_steps": self._forward_count,
                    "session_duration": time.time() - self._session_start_time if self._session_start_time else 0,
                    "session_status": "completed",
                    "model_class": self._model_class_name
                },
                "complete_conversation": complete_conversation,
                "step_by_step_inputs": self._accumulated_inputs,
                "step_by_step_outputs": self._accumulated_outputs,
                "all_forward_records": all_records,
                "summary": {
                    "total_tensor_records": sum(len(records) for records in all_records.values()),
                    "record_keys": list(all_records.keys()),
                    "successful_steps": len(self._accumulated_outputs),
                    "has_complete_conversation": len(complete_conversation) > 0,
                    "layer_statistics": self._compute_layer_statistics(all_records)
                }
            }
            
            timestamp = int(time.time())
            if self._model_class_name:
                filename = f"{self._model_class_name}_inference_session_{timestamp}.json"
            else:
                filename = f"inference_session_{timestamp}.json"
            
            debug_dir = "debug_outputs"
            os.makedirs(debug_dir, exist_ok=True)
            filepath = os.path.join(debug_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
            
            self._accumulated_inputs = []
            self._accumulated_outputs = []
            
            return filepath
            
        except Exception as e:
            print(f"Error saving complete inference session: {str(e)}")
            return None
    
    def force_save(self):
        if not self._inference_session_active:
            print("No active inference session")
            return None
        
        try:
            timestamp = int(time.time())
            if self._model_class_name:
                filename = f"{self._model_class_name}_forced_save_{timestamp}.json"
            else:
                filename = f"forced_save_{timestamp}.json"
            
            all_records = self.get_records()
            debug_info = {
                "session_info": {
                    "total_forward_steps": self._forward_count,
                    "session_duration": time.time() - self._session_start_time if self._session_start_time else 0,
                    "session_status": "forced_save_in_progress",
                    "model_class": self._model_class_name
                },
                "step_by_step_inputs": self._accumulated_inputs,
                "step_by_step_outputs": self._accumulated_outputs,
                "all_forward_records": all_records,
                "summary": {
                    "total_tensor_records": sum(len(records) for records in all_records.values()),
                    "record_keys": list(all_records.keys()),
                    "successful_steps": len(self._accumulated_outputs),
                }
            }
            
            debug_dir = "debug_outputs"
            os.makedirs(debug_dir, exist_ok=True)
            filepath = os.path.join(debug_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False, cls=TensorJSONEncoder)
            
            print(f"Forced save completed: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Forced save failed: {str(e)}")
            return None

    def print(self, tensor: Union[torch.Tensor, jnp.ndarray], name: str, stage: str = "", extra_info: str = ""):
        if not self.enabled or not self._inference_session_active:
            return
        
        if tensor is None:
            print(f"[{stage}] {name}: None")
            return
        
        key = f"{stage}_{name}" if stage else name
        
        if hasattr(tensor, 'cpu'): 
            stats = self._compute_pytorch_stats(tensor, name, stage, extra_info)
        else:  # JAX
            stats = self._compute_jax_stats(tensor, name, stage, extra_info)
        
        if self._inference_session_active:
            stats['forward_step'] = self._current_step
            stats['forward_count'] = self._forward_count
        
        with self.lock:
            if key not in self.records:
                self.records[key] = []
            self.records[key].append(stats)
        
        self._print_stats(stats, key)
    
    def _compute_pytorch_stats(self, tensor: torch.Tensor, name: str, stage: str, extra_info: str = "") -> Dict[str, Any]:
        if hasattr(tensor, 'cpu'):
            tensor_cpu = tensor.cpu()
        else:
            tensor_cpu = tensor
        
        try:
            # 处理整数类型，需要转换为浮点类型进行统计计算
            if tensor_cpu.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                tensor_for_stats = tensor_cpu.float()
            else:
                tensor_for_stats = tensor_cpu
            
            # 安全的标准差计算，避免单元素时的NaN
            # 统一使用总体标准差(unbiased=False)以与JAX保持一致
            if tensor_cpu.numel() > 1:
                std_val = float(tensor_for_stats.std(unbiased=False))
            else:
                std_val = 0.0
            
            stats = {
                'framework': 'pytorch',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'min': float(tensor_for_stats.min()),
                'max': float(tensor_for_stats.max()),
                'mean': float(tensor_for_stats.mean()),
                'std': std_val,
                'has_nan': bool(torch.any(torch.isnan(tensor_for_stats))),
                'has_inf': bool(torch.any(torch.isinf(tensor_for_stats))),
                'extra_info': extra_info
            }
            
            layer_id = 'unknown'
            module_type = 'unknown'
            
            # 改进的模块类型识别
            if '_layer_id_' in stage:
                parts = stage.split('_layer_id_')
                if len(parts) >= 2:
                    try:
                        layer_id = int(parts[1].split('_')[0])
                        module_type = parts[0]
                    except (ValueError, IndexError):
                        pass
            elif stage:
                stage_lower = stage.lower()
                if 'attention' in stage_lower:
                    module_type = 'attention'
                elif 'mlp' in stage_lower:
                    module_type = 'mlp'
                elif 'block' in stage_lower:
                    module_type = 'block'
                elif 'transformer' in stage_lower:
                    module_type = 'transformer'
                    layer_id = 'all'
                elif 'embed' in stage_lower:
                    module_type = 'embedding'
                    layer_id = 'all'
                elif 'rmsnorm' in stage_lower or 'layernorm' in stage_lower or 'norm' in stage_lower:
                    module_type = 'layernorm'
                    # 检查是否是final norm
                    if 'final' in stage_lower:
                        layer_id = 'all'
                elif 'lm_head' in stage_lower or 'logits' in stage_lower:
                    module_type = 'lm_head'
                    layer_id = 'all'
                
                # 提取layer_id（如果存在）
                import re
                layer_match = re.search(r'layer_id[_-](\d+)', stage, re.IGNORECASE)
                if layer_match:
                    try:
                        layer_id = int(layer_match.group(1))
                    except ValueError:
                        pass
            
            stats['layer_id'] = layer_id
            stats['module_type'] = module_type
            
        except Exception as e:
            stats = {
                'framework': 'pytorch',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'extra_info': extra_info,
                'layer_id': 'unknown',
                'module_type': 'unknown',
                'error': str(e)
            }
        
        return stats
    
    def _compute_jax_stats(self, tensor: jnp.ndarray, name: str, stage: str, extra_info: str) -> Dict[str, Any]:
        try:
            try:
                test_scalar = jnp.array(1.0)
                _ = test_scalar.item()
                can_concretize = True
            except Exception:
                can_concretize = False
            
            if can_concretize:
                if tensor.size > 1:
                    std_val = float(jnp.std(tensor, ddof=0).item())
                else:
                    std_val = 0.0
                
                stats = {
                    'framework': 'jax',
                    'name': name,
                    'stage': stage,
                    'shape': tuple(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'min': float(jnp.min(tensor).item()),
                    'max': float(jnp.max(tensor).item()),
                    'mean': float(jnp.mean(tensor).item()),
                    'std': std_val,
                    'has_nan': bool(jnp.any(jnp.isnan(tensor)).item()),
                    'has_inf': bool(jnp.any(jnp.isinf(tensor)).item()),
                    'extra_info': extra_info
                }
            else:
                stats = {
                    'framework': 'jax',
                    'name': name,
                    'stage': stage,
                    'shape': tuple(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'min': 'traced',
                    'max': 'traced', 
                    'mean': 'traced',
                    'std': 'traced',
                    'has_nan': 'traced',
                    'has_inf': 'traced',
                    'extra_info': extra_info,
                    'tracing_context': True
                }
            
            # 改进的layer_id提取逻辑
            layer_id = 'unknown'
            module_type = 'unknown'
            
            if '_layer_id_' in stage:
                parts = stage.split('_layer_id_')
                if len(parts) >= 2:
                    try:
                        layer_id = int(parts[1].split('_')[0])
                        module_type = parts[0]
                    except (ValueError, IndexError):
                        pass
            elif stage:
                stage_lower = stage.lower()
                if 'attention' in stage_lower:
                    module_type = 'attention'
                elif 'mlp' in stage_lower:
                    module_type = 'mlp'
                elif 'block' in stage_lower:
                    module_type = 'block'
                elif 'transformer' in stage_lower:
                    module_type = 'transformer'
                    layer_id = 'all'
                elif 'embed' in stage_lower:
                    module_type = 'embedding'
                    layer_id = 'all'
                elif 'rmsnorm' in stage_lower or 'layernorm' in stage_lower or 'norm' in stage_lower:
                    module_type = 'layernorm'
                    # 检查是否是final norm
                    if 'final' in stage_lower:
                        layer_id = 'all'
                elif 'lm_head' in stage_lower or 'logits' in stage_lower:
                    module_type = 'lm_head'
                    layer_id = 'all'
                
                # 提取layer_id（如果存在）
                import re
                layer_match = re.search(r'layer_id[_-](\d+)', stage, re.IGNORECASE)
                if layer_match:
                    try:
                        layer_id = int(layer_match.group(1))
                    except ValueError:
                        pass
            
            stats['layer_id'] = layer_id
            stats['module_type'] = module_type

        except Exception as e:
            stats = {
                'framework': 'jax',
                'name': name,
                'stage': stage,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'extra_info': extra_info,
                'layer_id': 'unknown',
                'module_type': 'unknown',
                'error': str(e)
            }
        
        return stats
    
    def _print_stats(self, stats: Dict[str, Any], key: str):
        if 'error' in stats:
            print(f"[{stats['stage']}] {stats['name']}: shape={stats['shape']}, dtype={stats['dtype']}, error={stats['error']}")
        elif stats.get('tracing_context', False):
            # 在JAX追踪上下文中的特殊处理
            framework = stats['framework'].upper()
            extra = f" {stats.get('extra_info', '')}" if stats.get('extra_info') else ""
            step_info = ""
            if 'forward_step' in stats:
                step_info = f"[Step {stats['forward_step']}]"
            
            print(f"{step_info}[{framework}][{stats['stage']}] {stats['name']}: shape={stats['shape']}, "
                  f"dtype={stats['dtype']}, TRACED_CONTEXT{extra}")
        else:
            framework = stats['framework'].upper()
            extra = f" {stats.get('extra_info', '')}" if stats.get('extra_info') else ""
            nan_inf = ""
            if stats['has_nan']:
                nan_inf += ", HAS_NAN"
            if stats['has_inf']:
                nan_inf += ", HAS_INF"
            
            # Add step information to the output
            step_info = ""
            if 'forward_step' in stats:
                step_info = f"[Step {stats['forward_step']}]"
            
            print(f"{step_info}[{framework}][{stats['stage']}] {stats['name']}: shape={stats['shape']}, "
                  f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
                  f"mean={stats['mean']:.6f}, std={stats['std']:.6f}{nan_inf}{extra}")
    
    def get_records(self, key: str = None) -> Union[Dict[str, List], List]:
        with self.lock:
            if key is None:
                return dict(self.records)
            return self.records.get(key, [])
    
    def clear_records(self):
        with self.lock:
            self.records.clear()
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def compare_frameworks(self, jax_key: str, pytorch_key: str, tolerance: float = 1e-5) -> bool:
        jax_records = self.get_records(jax_key)
        pytorch_records = self.get_records(pytorch_key)
        
        if not jax_records or not pytorch_records:
            print(f"Warning: record {jax_key} or {pytorch_key} is empty")
            return False
        
        print(f"\n=== compare {jax_key} (JAX) vs {pytorch_key} (PyTorch) ===")
        min_len = min(len(jax_records), len(pytorch_records))
        
        all_match = True
        for i in range(min_len):
            jax_record = jax_records[i]
            pytorch_record = pytorch_records[i]
            
            print(f"Step {i}:")
            print(f"  Shape: {jax_record['shape']} vs {pytorch_record['shape']}")
            
            if jax_record['shape'] != pytorch_record['shape']:
                print(f"  ❌ Shape mismatch!")
                all_match = False
                continue
            
            for metric in ['min', 'max', 'mean', 'std']:
                if metric in jax_record and metric in pytorch_record:
                    diff = abs(jax_record[metric] - pytorch_record[metric])
                    match = diff <= tolerance
                    status = "✅" if match else "❌"
                    print(f"  {metric.capitalize()}: {jax_record[metric]:.6f} vs {pytorch_record[metric]:.6f} "
                          f"(diff: {diff:.6f}) {status}")
                    if not match:
                        all_match = False
            
            for flag in ['has_nan', 'has_inf']:
                if flag in jax_record and flag in pytorch_record:
                    match = jax_record[flag] == pytorch_record[flag]
                    status = "✅" if match else "❌"
                    print(f"  {flag}: {jax_record[flag]} vs {pytorch_record[flag]} {status}")
                    if not match:
                        all_match = False
            
            print()
        
        return all_match
    
    def compare_records(self, other_records: List[Dict], tolerance: float = 1e-5) -> bool:
        all_records = []
        for records_list in self.records.values():
            all_records.extend(records_list)
        
        if len(all_records) != len(other_records):
            print(f"Record count mismatch: {len(all_records)} vs {len(other_records)}")
            return False
        
        all_match = True
        for i, (record1, record2) in enumerate(zip(all_records, other_records)):
            if record1['name'] != record2['name'] or record1['stage'] != record2['stage']:
                print(f"Record {i}: name/stage mismatch")
                all_match = False
                continue
            
            for key in ['min', 'max', 'mean', 'std']:
                if key in record1 and key in record2:
                    diff = abs(record1[key] - record2[key])
                    if diff > tolerance:
                        print(f"Record {i} ({record1['name']}): {key} differs by {diff:.8f}")
                        all_match = False
            
            for key in ['has_nan', 'has_inf']:
                if key in record1 and key in record2:
                    if record1[key] != record2[key]:
                        print(f"Record {i} ({record1['name']}): {key} differs")
                        all_match = False
        
        return all_match

    def _compute_layer_statistics(self, all_records: Dict) -> Dict[str, Any]:
        layer_stats = {}
        module_type_stats = {}
        step_stats = {}
        
        for record_key, records_list in all_records.items():
            for record in records_list:
                layer_id = record.get('layer_id', 'unknown')
                module_type = record.get('module_type', 'unknown')
                stage = record.get('stage', 'unknown')
                forward_step = record.get('forward_step', 'unknown')
                
                # Ensure forward_step is consistently typed for sorting
                if forward_step != 'unknown' and isinstance(forward_step, (int, float)):
                    forward_step = int(forward_step)
                
                if forward_step not in step_stats:
                    step_stats[forward_step] = {
                        'total_records': 0,
                        'layers': set(),
                        'modules': set(),
                        'tensor_count': 0
                    }
                
                step_stats[forward_step]['total_records'] += 1
                step_stats[forward_step]['layers'].add(layer_id)
                step_stats[forward_step]['modules'].add(module_type)
                step_stats[forward_step]['tensor_count'] += 1
                
                # Ensure layer_id is consistently typed
                if layer_id != 'unknown' and isinstance(layer_id, (int, float)):
                    layer_id = int(layer_id)
                
                if layer_id not in layer_stats:
                    layer_stats[layer_id] = {
                        'total_records': 0,
                        'modules': {},
                        'tensor_shapes': [],
                        'dtypes': set(),
                        'steps': set()
                    }
                
                layer_stats[layer_id]['total_records'] += 1
                layer_stats[layer_id]['tensor_shapes'].append(record.get('shape', []))
                layer_stats[layer_id]['dtypes'].add(record.get('dtype', 'unknown'))
                layer_stats[layer_id]['steps'].add(forward_step)
                
                if module_type not in layer_stats[layer_id]['modules']:
                    layer_stats[layer_id]['modules'][module_type] = 0
                layer_stats[layer_id]['modules'][module_type] += 1
                
                if module_type not in module_type_stats:
                    module_type_stats[module_type] = {
                        'total_records': 0,
                        'layers': set(),
                        'steps': set(),
                        'avg_tensor_size': 0,
                        'total_elements': 0
                    }
                
                module_type_stats[module_type]['total_records'] += 1
                module_type_stats[module_type]['layers'].add(layer_id)
                module_type_stats[module_type]['steps'].add(forward_step)
                
                shape = record.get('shape', [])
                if shape:
                    elements = 1
                    for dim in shape:
                        elements *= dim
                    module_type_stats[module_type]['total_elements'] += elements
        
        def safe_sort(items):
            try:
                numbers = [x for x in items if isinstance(x, (int, float))]
                strings = [x for x in items if isinstance(x, str)]
                return sorted(numbers) + sorted(strings)
            except Exception:
                return sorted([str(x) for x in items])
        
        for module_type, stats in module_type_stats.items():
            stats['layers'] = safe_sort(list(stats['layers']))
            stats['steps'] = safe_sort(list(stats['steps']))
            if stats['total_records'] > 0:
                stats['avg_tensor_size'] = stats['total_elements'] / stats['total_records']
        
        for layer_id, stats in layer_stats.items():
            stats['dtypes'] = list(stats['dtypes'])
            stats['steps'] = safe_sort(list(stats['steps']))
        
        for step, stats in step_stats.items():
            stats['layers'] = safe_sort(list(stats['layers']))
            stats['modules'] = safe_sort(list(stats['modules']))
        
        valid_layers = [k for k in layer_stats.keys() if k not in ['unknown']]
        valid_modules = [k for k in module_type_stats.keys() if k not in ['unknown']]
        valid_steps = [k for k in step_stats.keys() if k not in ['unknown']]
        
        return {
            'by_layer': layer_stats,
            'by_module_type': module_type_stats,
            'by_step': step_stats,
            'total_layers': len(valid_layers),
            'total_module_types': len(valid_modules),
            'total_steps': len(valid_steps)
        }


# Decorator for automatic function tracing
def trace_function(stage: str = "", include_args: bool = True, include_output: bool = True, context_fn: Optional[Callable] = None):
    """
    Decorator to automatically trace function inputs and outputs
    
    Args:
        stage: Stage name for the trace
        include_args: Whether to trace input arguments
        include_output: Whether to trace output
        context_fn: Optional function to extract context information (e.g., layer_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not global_tracer.enabled:
                return func(*args, **kwargs)
            
            func_name = func.__name__
            stage_name = stage or func_name.upper()
            
            context_info = {}
            if context_fn:
                try:
                    context_info = context_fn(*args, **kwargs)
                except Exception as e:
                    context_info = {"context_error": str(e)}
            
            if args and hasattr(args[0], '__class__'):
                self_obj = args[0]
                for attr_name in ['layer_id', 'layer_idx', 'id', 'idx']:
                    if hasattr(self_obj, attr_name):
                        context_info[attr_name] = getattr(self_obj, attr_name)
                        break
            
            if context_info:
                context_str = "_".join([f"{k}_{v}" for k, v in context_info.items() if k != "context_error"])
                if context_str:
                    stage_name = f"{stage_name}_{context_str}"
            
            if include_args:
                for i, arg in enumerate(args):
                    if hasattr(arg, 'shape'):
                        global_tracer.print(arg, f"{func_name}_input_{i}", stage_name)
                
                for key, value in kwargs.items():
                    if hasattr(value, 'shape'): 
                        global_tracer.print(value, f"{func_name}_input_{key}", stage_name)
            
            result = func(*args, **kwargs)
            
            if include_output:
                if hasattr(result, 'shape'): 
                    global_tracer.print(result, f"{func_name}_output", stage_name)
                elif isinstance(result, (tuple, list)): 
                    for i, item in enumerate(result):
                        if hasattr(item, 'shape'):
                            global_tracer.print(item, f"{func_name}_output_{i}", stage_name)
                elif hasattr(result, '__dict__'):
                    for attr_name in dir(result):
                        if not attr_name.startswith('_'):
                            attr_value = getattr(result, attr_name)
                            if hasattr(attr_value, 'shape'):
                                global_tracer.print(attr_value, f"{func_name}_output_{attr_name}", stage_name)
            
            return result
        return wrapper
    return decorator


global_tracer = UnifiedDebugTracer()