"""
闭环诊断系统集成模块
整合PATE-Net诊断、机理验证和LLM推理的完整闭环验证架构
"""

import os
import sys
import numpy as np
import torch
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

from models import LSTMClassificationModel
from mechanism_verifier import MechanismVerifier
from llm_reasoning_engine import LLMReasoningEngine

class ClosedLoopDiagnosticSystem:
    """闭环诊断系统主控制器"""

    def __init__(self,
                 model_path='models_saved/lstm_transfer_all_channels/best_lstm_transfer_full_model.pth',
                 use_mock_llm=False,
                 device=None):
        """
        初始化闭环诊断系统

        Args:
            model_path: 预训练PATE模型路径
            use_mock_llm: 是否使用Mock LLM（用于测试）
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.use_mock_llm = use_mock_llm

        # 故障类型映射
        self.fault_names = ['Normal', 'Inner', 'Outer', 'Ball']
        self.fault_names_cn = ['正常', '内圈故障', '外圈故障', '滚动体故障']

        print("初始化闭环诊断系统...")

        # 初始化三个核心模块
        self.pate_model = None
        self.mechanism_verifier = MechanismVerifier()
        self.llm_engine = LLMReasoningEngine(use_mock=use_mock_llm)

        # 加载PATE模型
        self._load_pate_model()

        print("✓ 闭环诊断系统初始化完成")

    def _load_pate_model(self):
        """加载现有的PATE/LSTM分类模型"""
        try:
            from models import create_lstm_classification_model

            # 创建模型结构
            self.pate_model = create_lstm_classification_model(
                signal_length=2048,
                feature_dim=256,
                hidden_dim=128,
                num_layers=2,
                num_classes=4
            ).to(self.device)

            # 加载权重
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.pate_model.load_state_dict(state_dict)
                self.pate_model.eval()
                print(f"✓ PATE模型加载成功: {self.model_path}")
            else:
                print(f"⚠ 模型文件不存在: {self.model_path}")
                print("使用随机初始化的模型进行测试")

        except Exception as e:
            print(f"⚠ PATE模型加载失败: {e}")
            print("将使用模拟诊断结果")
            self.pate_model = None

    def diagnose_single_signal(self, signal_data: np.ndarray, signal_id: str = "unknown") -> Dict[str, Any]:
        """
        对单个信号进行闭环诊断

        Args:
            signal_data: 输入信号数据 (shape: [signal_length,])
            signal_id: 信号标识符

        Returns:
            完整的闭环诊断结果
        """
        print(f"开始闭环诊断: {signal_id}")

        # 第一步：PATE模型初步诊断
        print("  步骤1: PATE模型诊断假设生成...")
        pate_evidence = self._get_pate_diagnosis(signal_data)

        # 第二步：机理验证
        print("  步骤2: 物理机理验证...")
        mechanism_evidence = self.mechanism_verifier.verify_hypothesis(signal_data, pate_evidence)

        # 第三步：LLM专家会诊
        print("  步骤3: LLM专家推理...")
        llm_diagnosis = self.llm_engine.expert_diagnosis(pate_evidence, mechanism_evidence)

        # 整合最终结果
        final_result = self._integrate_diagnosis_results(
            signal_id, pate_evidence, mechanism_evidence, llm_diagnosis
        )

        print(f"  ✓ 闭环诊断完成: {final_result['final_diagnosis']}")

        return final_result

    def _get_pate_diagnosis(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """获取PATE模型的诊断假设"""
        try:
            if self.pate_model is None:
                # 模拟PATE诊断结果
                return self._mock_pate_diagnosis(signal_data)

            # 准备输入数据
            if len(signal_data.shape) == 1:
                signal_tensor = torch.FloatTensor(signal_data).unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                signal_tensor = torch.FloatTensor(signal_data).unsqueeze(0).to(self.device)

            # 模型推理
            with torch.no_grad():
                features, logits = self.pate_model(signal_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

                predicted_class = predicted_class.item()
                confidence = confidence.item()

                # 计算到原型的距离（模拟）
                prototype_distance = 1.0 - confidence

            return {
                'fault_type': self.fault_names[predicted_class],
                'fault_type_cn': self.fault_names_cn[predicted_class],
                'confidence': confidence,
                'distance_to_prototype': prototype_distance,
                'class_probabilities': probabilities.cpu().numpy().flatten().tolist(),
                'model_type': 'LSTM_Classification'
            }

        except Exception as e:
            print(f"PATE诊断出错: {e}")
            return self._mock_pate_diagnosis(signal_data)

    def _mock_pate_diagnosis(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """模拟PATE诊断（用于测试）"""
        # 基于信号统计特征的简单启发式诊断
        rms = np.sqrt(np.mean(signal_data**2))
        std = np.std(signal_data)
        kurtosis = np.mean((signal_data - np.mean(signal_data))**4) / (std**4)

        # 简单的故障判断逻辑
        if kurtosis > 5.0:
            fault_type = 'Inner'
            confidence = 0.85
        elif rms > np.percentile(signal_data, 95):
            fault_type = 'Outer'
            confidence = 0.78
        elif std > np.mean(np.abs(signal_data)) * 1.5:
            fault_type = 'Ball'
            confidence = 0.72
        else:
            fault_type = 'Normal'
            confidence = 0.90

        fault_idx = self.fault_names.index(fault_type)

        # 模拟概率分布
        probs = [0.1, 0.1, 0.1, 0.1]
        probs[fault_idx] = confidence
        remaining = (1.0 - confidence) / 3
        for i in range(4):
            if i != fault_idx:
                probs[i] = remaining

        return {
            'fault_type': fault_type,
            'fault_type_cn': self.fault_names_cn[fault_idx],
            'confidence': confidence,
            'distance_to_prototype': 1.0 - confidence,
            'class_probabilities': probs,
            'model_type': 'Mock_PATE'
        }

    def _integrate_diagnosis_results(self, signal_id: str, pate_evidence: Dict,
                                   mechanism_evidence: Dict, llm_diagnosis: Dict) -> Dict[str, Any]:
        """整合所有诊断结果"""

        # 确定最终诊断结果
        final_fault_type = llm_diagnosis.get('final_conclusion', pate_evidence['fault_type'])

        # 标准化诊断结果
        if final_fault_type in ['Uncertain', 'Error']:
            final_fault_type = pate_evidence['fault_type']  # 回退到PATE诊断

        # 获取中文名称
        try:
            final_fault_idx = self.fault_names.index(final_fault_type)
            final_fault_cn = self.fault_names_cn[final_fault_idx]
        except ValueError:
            final_fault_type = 'Normal'
            final_fault_cn = '正常'

        return {
            'signal_id': signal_id,
            'final_diagnosis': final_fault_type,
            'final_diagnosis_cn': final_fault_cn,
            'confidence_level': llm_diagnosis.get('confidence_level', '中等置信度'),
            'evidence_consistency': llm_diagnosis.get('evidence_consistency', '未知'),
            'reasoning_process': llm_diagnosis.get('reasoning_process', ''),

            # 详细证据链
            'evidence_chain': {
                'step1_pate_diagnosis': pate_evidence,
                'step2_mechanism_verification': mechanism_evidence,
                'step3_llm_reasoning': llm_diagnosis
            },

            # 诊断质量评估
            'diagnosis_quality': {
                'pate_confidence': pate_evidence['confidence'],
                'mechanism_confirmation': mechanism_evidence.get('verification_result', 'UNKNOWN'),
                'evidence_agreement': llm_diagnosis.get('evidence_consistency', '未知'),
                'overall_reliability': self._assess_overall_reliability(pate_evidence, mechanism_evidence, llm_diagnosis)
            }
        }

    def _assess_overall_reliability(self, pate_evidence: Dict, mechanism_evidence: Dict, llm_diagnosis: Dict) -> str:
        """评估整体诊断可靠性"""
        pate_conf = pate_evidence['confidence']
        mechanism_result = mechanism_evidence.get('verification_result', 'NO_CONFIRMATION')
        llm_confidence = llm_diagnosis.get('confidence_level', '低置信度')

        if (pate_conf > 0.9 and
            mechanism_result == 'STRONG_CONFIRMATION' and
            '高置信度' in llm_confidence):
            return 'VERY_HIGH'
        elif (pate_conf > 0.8 and
              mechanism_result in ['STRONG_CONFIRMATION', 'WEAK_CONFIRMATION']):
            return 'HIGH'
        elif pate_conf > 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'

    def diagnose_multiple_signals(self, signals_data: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        批量诊断多个信号

        Args:
            signals_data: [(signal_data, signal_id), ...] 信号数据和ID的列表

        Returns:
            诊断结果列表
        """
        print(f"开始批量闭环诊断: {len(signals_data)} 个信号")

        results = []
        for i, (signal_data, signal_id) in enumerate(signals_data):
            print(f"\n进度: {i+1}/{len(signals_data)}")
            try:
                result = self.diagnose_single_signal(signal_data, signal_id)
                results.append(result)
            except Exception as e:
                print(f"诊断失败 {signal_id}: {e}")
                # 创建错误结果
                error_result = {
                    'signal_id': signal_id,
                    'final_diagnosis': 'Error',
                    'final_diagnosis_cn': '诊断错误',
                    'error_message': str(e)
                }
                results.append(error_result)

        print(f"\n✓ 批量诊断完成: {len(results)} 个结果")
        return results

    def save_diagnosis_results(self, results: List[Dict], output_path: str):
        """保存诊断结果到文件"""
        try:
            # 保存详细JSON结果
            json_path = output_path.replace('.csv', '_detailed.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 保存简化CSV结果
            import pandas as pd

            csv_data = []
            for result in results:
                csv_data.append({
                    'Signal_ID': result['signal_id'],
                    'Final_Diagnosis': result['final_diagnosis'],
                    'Final_Diagnosis_CN': result.get('final_diagnosis_cn', ''),
                    'Confidence_Level': result.get('confidence_level', ''),
                    'PATE_Confidence': result.get('evidence_chain', {}).get('step1_pate_diagnosis', {}).get('confidence', 0),
                    'Mechanism_Result': result.get('evidence_chain', {}).get('step2_mechanism_verification', {}).get('verification_result', ''),
                    'Overall_Reliability': result.get('diagnosis_quality', {}).get('overall_reliability', ''),
                    'Reasoning': result.get('reasoning_process', '')
                })

            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False, encoding='utf-8')

            print(f"✓ 诊断结果已保存:")
            print(f"  详细结果: {json_path}")
            print(f"  简化结果: {csv_path}")

        except Exception as e:
            print(f"保存结果失败: {e}")

def test_closed_loop_system():
    """测试闭环诊断系统"""
    print("测试闭环诊断系统...")

    # 创建系统实例（使用mock模式）
    system = ClosedLoopDiagnosticSystem(use_mock_llm=True)

    # 生成测试信号
    fs = 32000
    t = np.linspace(0, 1, fs)

    # 测试信号1: 模拟内圈故障
    noise1 = np.random.normal(0, 0.1, len(t))
    bpfi = 54  # 内圈故障频率
    fault_signal1 = 0.5 * np.sin(2 * np.pi * bpfi * t) * np.exp(-10 * (t % (1/bpfi)))
    test_signal1 = noise1 + fault_signal1

    # 测试信号2: 正常信号
    test_signal2 = np.random.normal(0, 0.05, len(t))

    # 单个信号诊断测试
    print("\n=== 单个信号诊断测试 ===")
    result1 = system.diagnose_single_signal(test_signal1, "test_inner_fault")
    print(f"诊断结果: {result1['final_diagnosis']}")
    print(f"置信度: {result1['confidence_level']}")

    # 批量诊断测试
    print("\n=== 批量诊断测试 ===")
    signals_batch = [
        (test_signal1, "signal_1_inner"),
        (test_signal2, "signal_2_normal")
    ]

    batch_results = system.diagnose_multiple_signals(signals_batch)

    for result in batch_results:
        print(f"{result['signal_id']}: {result['final_diagnosis']} ({result.get('confidence_level', 'N/A')})")

    print("\n✓ 闭环诊断系统测试完成!")

    return batch_results

if __name__ == '__main__':
    test_closed_loop_system()