"""
多通道目标域数据预测脚本
对16个目标域数据文件(A-P)进行故障诊断预测
使用训练好的LSTM迁移学习模型
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import resample
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import create_lstm_classification_model

class TargetDomainPredictor:
    """目标域数据预测器"""

    def __init__(self, model_path, target_sr=32000, segment_length=2048, overlap_ratio=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.overlap_ratio = overlap_ratio
        self.window_step = int(segment_length * (1 - overlap_ratio))

        # 故障类别映射
        self.fault_names = {
            0: 'Normal',
            1: 'Inner',
            2: 'Outer',
            3: 'Ball'
        }

        self.fault_names_cn = {
            0: '正常',
            1: '内圈故障',
            2: '外圈故障',
            3: '滚动体故障'
        }

        print(f"使用设备: {self.device}")

        # 加载模型
        self.load_model(model_path)

    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")

        # 创建模型架构
        self.model = create_lstm_classification_model(
            signal_length=self.segment_length,
            feature_dim=256,
            hidden_dim=128,
            num_layers=2,
            num_classes=4
        ).to(self.device)

        # 加载权重
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✓ 模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

    def load_target_data(self, file_path):
        """加载目标域.mat文件"""
        try:
            data = loadmat(file_path)

            # 提取数据（目标域数据可能只有一个通道）
            signal_data = None
            for key in data.keys():
                if not key.startswith('__') and isinstance(data[key], np.ndarray):
                    if data[key].ndim == 1 or (data[key].ndim == 2 and min(data[key].shape) == 1):
                        signal_data = data[key].flatten()
                        break

            if signal_data is None:
                raise ValueError(f"无法从文件 {file_path} 中提取信号数据")

            print(f"加载文件 {os.path.basename(file_path)}: {len(signal_data)} 个采样点")
            return signal_data

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return None

    def preprocess_signal(self, signal):
        """预处理信号：重采样和切片"""
        # 如果信号采样率不是32kHz，进行重采样
        original_length = len(signal)
        if original_length != int(8 * self.target_sr):  # 目标域数据是8秒
            # 重采样到32kHz
            target_length = int(8 * self.target_sr)  # 8秒 * 32000Hz = 256000
            signal = resample(signal, target_length)
            print(f"重采样: {original_length} -> {len(signal)} 采样点")

        # 归一化
        signal = signal.astype(np.float32)
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            signal = (signal - mean) / std

        # 分段
        segments = []
        start = 0
        while start + self.segment_length <= len(signal):
            segment = signal[start:start + self.segment_length]
            segments.append(segment)
            start += self.window_step

        print(f"生成 {len(segments)} 个分段")
        return np.array(segments)

    def predict_segments(self, segments):
        """预测所有分段"""
        predictions = []
        probabilities = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(segments), 32):  # 批处理
                batch = segments[i:i+32]
                batch_tensor = torch.FloatTensor(batch).to(self.device)

                # 如果只有一个维度，需要添加通道维度
                if len(batch_tensor.shape) == 2:
                    batch_tensor = batch_tensor.unsqueeze(1)  # (batch, 1, length)

                # 预测
                features, logits = self.model(batch_tensor)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return np.array(predictions), np.array(probabilities)

    def aggregate_predictions(self, predictions, probabilities):
        """聚合预测结果"""
        # 多数投票
        vote_counts = Counter(predictions)
        majority_vote = vote_counts.most_common(1)[0][0]

        # 平均概率
        mean_probs = np.mean(probabilities, axis=0)
        prob_prediction = np.argmax(mean_probs)

        # 计算置信度统计
        confidence_stats = {
            'mean_confidence': np.mean(np.max(probabilities, axis=1)),
            'std_confidence': np.std(np.max(probabilities, axis=1)),
            'min_confidence': np.min(np.max(probabilities, axis=1)),
            'max_confidence': np.max(np.max(probabilities, axis=1))
        }

        return {
            'majority_vote': majority_vote,
            'probability_vote': prob_prediction,
            'vote_counts': dict(vote_counts),
            'mean_probabilities': mean_probs,
            'confidence_stats': confidence_stats,
            'segment_count': len(predictions)
        }

    def predict_file(self, file_path):
        """预测单个文件"""
        print(f"\n--- 预测文件: {os.path.basename(file_path)} ---")

        # 加载数据
        signal = self.load_target_data(file_path)
        if signal is None:
            return None

        # 预处理
        segments = self.preprocess_signal(signal)
        if len(segments) == 0:
            print("无法生成有效分段")
            return None

        # 预测
        predictions, probabilities = self.predict_segments(segments)

        # 聚合结果
        result = self.aggregate_predictions(predictions, probabilities)

        # 输出结果
        final_prediction = result['majority_vote']
        confidence = result['confidence_stats']['mean_confidence']

        print(f"预测结果: {self.fault_names[final_prediction]} ({self.fault_names_cn[final_prediction]})")
        print(f"平均置信度: {confidence:.4f}")
        print(f"投票分布: {result['vote_counts']}")
        print(f"各类别概率: {result['mean_probabilities']}")

        return {
            'file_name': os.path.basename(file_path),
            'predicted_label': final_prediction,
            'predicted_fault': self.fault_names[final_prediction],
            'predicted_fault_cn': self.fault_names_cn[final_prediction],
            'confidence': confidence,
            'vote_counts': result['vote_counts'],
            'mean_probabilities': result['mean_probabilities'].tolist(),
            'segment_count': result['segment_count']
        }

    def predict_target_domain(self, target_dir):
        """预测整个目标域"""
        print("=== 开始目标域预测 ===")
        print(f"目标目录: {target_dir}")

        # 获取所有.mat文件
        target_files = []
        for file_name in os.listdir(target_dir):
            if file_name.endswith('.mat'):
                target_files.append(os.path.join(target_dir, file_name))

        target_files.sort()  # 按文件名排序
        print(f"找到 {len(target_files)} 个目标域文件")

        # 预测所有文件
        results = []
        for file_path in target_files:
            result = self.predict_file(file_path)
            if result:
                results.append(result)

        # 保存结果
        self.save_results(results)

        # 显示汇总
        self.print_summary(results)

        return results

    def save_results(self, results):
        """保存预测结果"""
        if not results:
            print("没有有效的预测结果")
            return

        # 创建DataFrame
        df_data = []
        for result in results:
            df_data.append({
                'File_Name': result['file_name'],
                'Predicted_Label': result['predicted_label'],
                'Predicted_Fault_EN': result['predicted_fault'],
                'Predicted_Fault_CN': result['predicted_fault_cn'],
                'Confidence': result['confidence'],
                'Segment_Count': result['segment_count'],
                'Prob_Normal': result['mean_probabilities'][0],
                'Prob_Inner': result['mean_probabilities'][1],
                'Prob_Outer': result['mean_probabilities'][2],
                'Prob_Ball': result['mean_probabilities'][3]
            })

        df = pd.DataFrame(df_data)

        # 保存到CSV
        output_path = 'results/target_domain_predictions_all_channels.csv'
        os.makedirs('results', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ 预测结果保存到: {output_path}")

        # 简化JSON保存，只保存主要信息
        import json
        simple_results = []
        for result in results:
            simple_result = {
                'file_name': result['file_name'],
                'predicted_label': int(result['predicted_label']),
                'predicted_fault': result['predicted_fault'],
                'predicted_fault_cn': result['predicted_fault_cn'],
                'confidence': float(result['confidence']),
                'segment_count': int(result['segment_count'])
            }
            simple_results.append(simple_result)

        json_path = 'results/target_domain_predictions_detailed_all_channels.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simple_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 详细结果保存到: {json_path}")

    def print_summary(self, results):
        """打印预测汇总"""
        if not results:
            return

        print("\n" + "="*60)
        print("目标域预测汇总")
        print("="*60)

        # 统计各类别预测数量
        label_counts = Counter([r['predicted_label'] for r in results])

        print(f"总文件数: {len(results)}")
        print("\n预测分布:")
        for label, count in sorted(label_counts.items()):
            fault_name = self.fault_names[label]
            fault_name_cn = self.fault_names_cn[label]
            percentage = count / len(results) * 100
            print(f"  {fault_name} ({fault_name_cn}): {count} 个文件 ({percentage:.1f}%)")

        # 平均置信度
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\n平均预测置信度: {avg_confidence:.4f}")

        # 详细列表
        print(f"\n详细预测结果:")
        print(f"{'文件':<8} {'预测标签':<12} {'预测结果':<15} {'置信度':<8}")
        print("-" * 50)
        for result in results:
            print(f"{result['file_name']:<8} "
                  f"{result['predicted_label']:<12} "
                  f"{result['predicted_fault_cn']:<15} "
                  f"{result['confidence']:<8.4f}")

def main():
    # 配置参数
    model_path = 'models_saved/lstm_transfer_all_channels/best_lstm_transfer_full_model.pth'
    target_dir = 'data/target_domain'

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先训练模型!")
        return

    # 检查目标目录
    if not os.path.exists(target_dir):
        print(f"错误: 目标目录不存在 {target_dir}")
        return

    # 创建预测器
    predictor = TargetDomainPredictor(model_path)

    # 执行预测
    results = predictor.predict_target_domain(target_dir)

    print("\n=== 目标域预测完成 ===")

if __name__ == '__main__':
    main()