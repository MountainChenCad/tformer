"""
LLM推理引擎 - 基于大语言模型的专家会诊系统
实现LLM驱动的专家推理，整合深度学习诊断和物理机理验证的双重证据
"""

import os
import json
import re
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using mock LLM engine")

class LLMReasoningEngine:
    """LLM专家推理引擎"""

    def __init__(self, model_path="/autodl-tmp/LLM/Qwen2.5-0.5B-Instruct", use_mock=False):
        """
        初始化LLM推理引擎

        Args:
            model_path: 模型路径，如果为None则尝试自动查找或使用mock
            use_mock: 是否使用模拟模式（用于测试）
        """
        self.use_mock = use_mock or not TRANSFORMERS_AVAILABLE
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TRANSFORMERS_AVAILABLE else None

        if not self.use_mock:
            self._load_model()
        else:
            print("Using Mock LLM Engine for testing")

    def _load_model(self):
        """加载LLM模型"""
        try:
            # 尝试多个可能的模型路径
            possible_paths = [
                self.model_path,
                "/autodl-tmp/LLM/Qwen2.5-0.5B-Instruct",
                "/root/models/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-0.5B-Instruct",  # HuggingFace hub
            ]

            model_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    model_path = path
                    break

            if not model_path:
                # 尝试从HuggingFace下载
                model_path = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"Attempting to download model from HuggingFace: {model_path}")

            # 加载tokenizer和model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

            print(f"✓ LLM model loaded successfully from: {model_path}")

        except Exception as e:
            print(f"Failed to load LLM model: {e}")
            print("Falling back to Mock LLM Engine")
            self.use_mock = True

    def expert_diagnosis(self, pate_evidence: Dict[str, Any], mechanism_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行专家会诊：基于双重证据进行LLM推理

        Args:
            pate_evidence: PATE模型的诊断证据
            mechanism_evidence: 机理验证模块的证据

        Returns:
            final_diagnosis: 包含最终诊断结论和推理过程的字典
        """
        # 构建专家会诊的prompt
        expert_prompt = self._construct_expert_prompt(pate_evidence, mechanism_evidence)

        # 生成LLM推理结果
        if self.use_mock:
            response = self._mock_llm_inference(pate_evidence, mechanism_evidence)
        else:
            response = self._generate_llm_response(expert_prompt)

        # 解析LLM响应并结构化输出
        diagnosis_result = self._parse_diagnosis_response(response, pate_evidence, mechanism_evidence)

        return diagnosis_result

    def _construct_expert_prompt(self, pate_evidence: Dict, mechanism_evidence: Dict) -> str:
        """构建专家会诊的结构化prompt"""

        # 从机理证据中提取理论频率信息
        theoretical_freqs = mechanism_evidence.get('analysis_parameters', {}).get('theoretical_frequencies', {})

        prompt = f"""# 角色：世界顶级机械故障诊断专家

你是一名拥有深厚信号处理知识和设备动力学背景的机械故障诊断专家。你正在主持一个由深度学习AI和传统信号分析工具组成的联合诊断会诊。

## 诊断对象：高速列车轴承
- **轴承转速**: 约600 RPM
- **内圈故障特征频率(BPFI)**: {theoretical_freqs.get('BPFI', 'N/A')} Hz
- **外圈故障特征频率(BPFO)**: {theoretical_freqs.get('BPFO', 'N/A')} Hz
- **滚动体故障特征频率(BSF)**: {theoretical_freqs.get('BSF', 'N/A')} Hz

## 证据清单

### 证据A：来自深度学习模型(PATE-Net)的初步诊断
- **诊断假设**: {pate_evidence.get('fault_type', '未知')}
- **模型置信度**: {pate_evidence.get('confidence', 0.0):.3f}
- **内部依据**: 模型认为该信号特征与"{pate_evidence.get('fault_type', '未知')}"的标准原型最匹配

### 证据B：来自物理机理验证模块的定向分析结果
- **验证目标**: {mechanism_evidence.get('target_fault_type', 'N/A')}
- **分析方法**: {mechanism_evidence.get('analysis_type', 'N/A')}
- **检查频率点**: {mechanism_evidence.get('theoretical_frequency_hz', 'N/A')} Hz
- **分析结果**:
    - **是否发现预期峰值**: {mechanism_evidence.get('peak_found', False)}
    - **峰值显著性**: {mechanism_evidence.get('peak_prominence_db', 0)} dB
    - **谐波数量**: {mechanism_evidence.get('harmonics_detected', 0)}
    - **验证结果**: {mechanism_evidence.get('verification_result', 'N/A')}

## 任务：基于以上两份独立来源的证据，进行专家决策

请完成以下分析：

1. **证据一致性分析**: 判断证据A和证据B是否相互支持。如果存在矛盾，请明确指出矛盾点。

2. **最终诊断结论**: 给出你综合判断后的最终故障类型。

3. **诊断置信度评级**: 基于证据的一致性和强度，将你的最终诊断信心评为：[高置信度 | 中等置信度 | 低置信度/证据冲突]

4. **诊断推理过程**: 详细解释你做出此判断的逻辑链条。

请严格按照以下格式输出：

**证据一致性**: [一致/部分一致/冲突]
**最终诊断**: [Normal/Inner/Outer/Ball]
**置信度等级**: [高置信度/中等置信度/低置信度]
**推理过程**: [详细的推理分析]
"""

        return prompt

    def _generate_llm_response(self, prompt: str) -> str:
        """使用真实LLM模型生成响应"""
        try:
            messages = [
                {"role": "system", "content": "你是一名专业的机械故障诊断专家。"},
                {"role": "user", "content": prompt}
            ]

            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
            )

            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码响应
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response

        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return self._mock_llm_inference({"fault_type": "Unknown"}, {"verification_result": "ERROR"})

    def _mock_llm_inference(self, pate_evidence: Dict, mechanism_evidence: Dict) -> str:
        """模拟LLM推理（用于测试）"""
        pate_fault = pate_evidence.get('fault_type', 'Unknown')
        pate_confidence = pate_evidence.get('confidence', 0.0)
        mechanism_result = mechanism_evidence.get('verification_result', 'NO_CONFIRMATION')
        peak_found = mechanism_evidence.get('peak_found', False)
        prominence = mechanism_evidence.get('peak_prominence_db', 0.0)

        # 模拟专家推理逻辑
        if pate_fault == 'Normal':
            consistency = "一致"
            final_diagnosis = "Normal"
            confidence_level = "高置信度"
            reasoning = "深度学习模型诊断为正常状态，无需进行机理验证。两者结论一致。"

        elif mechanism_result == 'STRONG_CONFIRMATION':
            consistency = "一致"
            final_diagnosis = pate_fault
            confidence_level = "高置信度"
            reasoning = f"深度学习模型诊断为{pate_fault}故障（置信度{pate_confidence:.3f}），机理验证在理论频率处发现强烈峰值（显著性{prominence:.1f}dB），两证据完美印证，形成可靠的诊断闭环。"

        elif mechanism_result == 'WEAK_CONFIRMATION':
            consistency = "部分一致"
            final_diagnosis = pate_fault
            confidence_level = "中等置信度"
            reasoning = f"深度学习模型诊断为{pate_fault}故障，机理验证发现较弱的确认证据，建议结合其他分析方法进一步确认。"

        elif mechanism_result == 'NO_CONFIRMATION' and peak_found == False:
            consistency = "冲突"
            final_diagnosis = "Uncertain"
            confidence_level = "低置信度"
            reasoning = f"深度学习模型高度怀疑是{pate_fault}故障，但机理验证未能在理论频率处找到相应证据。可能原因：1)故障处于早期阶段；2)信号噪声干扰；3)模型误判。建议进行进一步分析。"

        else:
            consistency = "部分一致"
            final_diagnosis = pate_fault
            confidence_level = "中等置信度"
            reasoning = f"基于现有证据，倾向于{pate_fault}故障诊断，但需要更多证据支持。"

        mock_response = f"""**证据一致性**: {consistency}
**最终诊断**: {final_diagnosis}
**置信度等级**: {confidence_level}
**推理过程**: {reasoning}"""

        return mock_response

    def _parse_diagnosis_response(self, response: str, pate_evidence: Dict, mechanism_evidence: Dict) -> Dict[str, Any]:
        """解析LLM响应并提取结构化诊断结果"""
        try:
            # 使用正则表达式提取关键信息
            consistency_match = re.search(r'\*\*证据一致性\*\*:?\s*([^\n*]+)', response)
            diagnosis_match = re.search(r'\*\*最终诊断\*\*:?\s*([^\n*]+)', response)
            confidence_match = re.search(r'\*\*置信度等级\*\*:?\s*([^\n*]+)', response)
            reasoning_match = re.search(r'\*\*推理过程\*\*:?\s*([^\n*]+)', response)

            consistency = consistency_match.group(1).strip() if consistency_match else "未知"
            final_diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "Uncertain"
            confidence_level = confidence_match.group(1).strip() if confidence_match else "低置信度"
            reasoning_process = reasoning_match.group(1).strip() if reasoning_match else response.strip()

            # 标准化诊断结果
            if final_diagnosis.lower() in ['normal', '正常']:
                final_diagnosis = 'Normal'
            elif final_diagnosis.lower() in ['inner', '内圈', 'inner_race_fault']:
                final_diagnosis = 'Inner'
            elif final_diagnosis.lower() in ['outer', '外圈', 'outer_race_fault']:
                final_diagnosis = 'Outer'
            elif final_diagnosis.lower() in ['ball', '滚动体', 'ball_fault']:
                final_diagnosis = 'Ball'

            return {
                'final_conclusion': final_diagnosis,
                'confidence_level': confidence_level,
                'evidence_consistency': consistency,
                'reasoning_process': reasoning_process,
                'raw_llm_response': response,
                'input_evidence': {
                    'pate_evidence': pate_evidence,
                    'mechanism_evidence': mechanism_evidence
                }
            }

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                'final_conclusion': 'Error',
                'confidence_level': '无法确定',
                'evidence_consistency': '解析错误',
                'reasoning_process': f'LLM响应解析失败: {str(e)}',
                'raw_llm_response': response,
                'input_evidence': {'pate_evidence': pate_evidence, 'mechanism_evidence': mechanism_evidence}
            }

def test_llm_reasoning_engine():
    """测试LLM推理引擎"""
    print("Testing LLM Reasoning Engine...")

    # 创建推理引擎（使用mock模式进行测试）
    engine = LLMReasoningEngine(use_mock=True)

    # 测试案例1：证据一致
    print("\n=== 测试案例1：证据一致 ===")
    pate_evidence1 = {
        'fault_type': 'Inner',
        'confidence': 0.92,
        'distance_to_prototype': 0.15
    }

    mechanism_evidence1 = {
        'analysis_type': 'Envelope_Spectrum_Analysis',
        'target_fault_type': 'Inner',
        'theoretical_frequency_hz': 54.0,
        'peak_found': True,
        'peak_prominence_db': 15.2,
        'harmonics_detected': 3,
        'verification_result': 'STRONG_CONFIRMATION'
    }

    result1 = engine.expert_diagnosis(pate_evidence1, mechanism_evidence1)
    print(f"最终诊断: {result1['final_conclusion']}")
    print(f"置信度: {result1['confidence_level']}")
    print(f"推理过程: {result1['reasoning_process']}")

    # 测试案例2：证据冲突
    print("\n=== 测试案例2：证据冲突 ===")
    pate_evidence2 = {
        'fault_type': 'Outer',
        'confidence': 0.85,
        'distance_to_prototype': 0.22
    }

    mechanism_evidence2 = {
        'analysis_type': 'Envelope_Spectrum_Analysis',
        'target_fault_type': 'Outer',
        'theoretical_frequency_hz': 36.0,
        'peak_found': False,
        'peak_prominence_db': 3.1,
        'harmonics_detected': 0,
        'verification_result': 'NO_CONFIRMATION'
    }

    result2 = engine.expert_diagnosis(pate_evidence2, mechanism_evidence2)
    print(f"最终诊断: {result2['final_conclusion']}")
    print(f"置信度: {result2['confidence_level']}")
    print(f"推理过程: {result2['reasoning_process']}")

    print("\nLLM Reasoning Engine test completed!")

if __name__ == '__main__':
    test_llm_reasoning_engine()