"""
闭环预测脚本 - 对16个目标域文件进行闭环诊断
集成PATE-Net + 机理验证 + LLM推理的完整流程
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

from closed_loop_diagnostic_system import ClosedLoopDiagnosticSystem

def load_and_preprocess_target_file(file_path, target_length=256000):
    """
    加载并预处理目标域.mat文件

    Args:
        file_path: .mat文件路径
        target_length: 目标信号长度 (8秒 * 32kHz = 256000)

    Returns:
        preprocessed_signal: 预处理后的信号
    """
    try:
        # 加载.mat文件
        data = loadmat(file_path)

        # 查找信号数据
        signal_data = None
        for key in data.keys():
            if not key.startswith('__') and isinstance(data[key], np.ndarray):
                if data[key].ndim == 1 or (data[key].ndim == 2 and min(data[key].shape) == 1):
                    signal_data = data[key].flatten()
                    break

        if signal_data is None:
            raise ValueError("未找到有效的信号数据")

        # 重采样到目标长度
        if len(signal_data) != target_length:
            signal_data = resample(signal_data, target_length)

        # 归一化
        signal_data = signal_data.astype(np.float32)
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std > 0:
            signal_data = (signal_data - mean) / std

        return signal_data

    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None

def create_signal_segments(signal, segment_length=2048, overlap_ratio=0.5):
    """
    将长信号切分成多个段进行诊断

    Args:
        signal: 输入信号
        segment_length: 段长度
        overlap_ratio: 重叠比例

    Returns:
        segments: 信号段列表
    """
    step = int(segment_length * (1 - overlap_ratio))
    segments = []

    for start in range(0, len(signal) - segment_length + 1, step):
        segment = signal[start:start + segment_length]
        segments.append(segment)

    return segments

def aggregate_segment_results(segment_results):
    """
    聚合多个段的诊断结果

    Args:
        segment_results: 段诊断结果列表

    Returns:
        aggregated_result: 聚合后的诊断结果
    """
    if not segment_results:
        return None

    # 统计各类故障的投票
    fault_votes = {}
    confidence_scores = []
    reliability_scores = []

    for result in segment_results:
        fault_type = result['final_diagnosis']
        confidence_level = result.get('confidence_level', '中等置信度')

        if fault_type not in fault_votes:
            fault_votes[fault_type] = 0
        fault_votes[fault_type] += 1

        # 收集置信度信息
        pate_conf = result.get('evidence_chain', {}).get('step1_pate_diagnosis', {}).get('confidence', 0.5)
        confidence_scores.append(pate_conf)

        # 收集可靠性信息
        reliability = result.get('diagnosis_quality', {}).get('overall_reliability', 'MEDIUM')
        reliability_map = {'VERY_HIGH': 1.0, 'HIGH': 0.8, 'MEDIUM': 0.6, 'LOW': 0.4}
        reliability_scores.append(reliability_map.get(reliability, 0.5))

    # 多数投票确定最终诊断
    final_fault = max(fault_votes.items(), key=lambda x: x[1])[0]
    vote_ratio = fault_votes[final_fault] / len(segment_results)

    # 计算平均置信度
    avg_confidence = np.mean(confidence_scores)
    avg_reliability = np.mean(reliability_scores)

    # 确定整体置信度等级
    if vote_ratio >= 0.8 and avg_confidence >= 0.9 and avg_reliability >= 0.8:
        overall_confidence = '高置信度'
    elif vote_ratio >= 0.6 and avg_confidence >= 0.7:
        overall_confidence = '中等置信度'
    else:
        overall_confidence = '低置信度'

    # 获取代表性推理过程
    representative_result = segment_results[0]
    reasoning = representative_result.get('reasoning_process', '')

    # 构建聚合结果
    aggregated_result = {
        'final_diagnosis': final_fault,
        'confidence_level': overall_confidence,
        'segment_count': len(segment_results),
        'vote_ratio': round(vote_ratio, 3),
        'average_pate_confidence': round(avg_confidence, 3),
        'average_reliability': round(avg_reliability, 3),
        'fault_vote_distribution': fault_votes,
        'representative_reasoning': reasoning,
        'segment_results': segment_results  # 保留详细的段结果
    }

    return aggregated_result

def predict_target_domain_closed_loop(data_dir='data/target_domain',
                                    model_path='models_saved/lstm_transfer_all_channels/best_lstm_transfer_full_model.pth',
                                    output_dir='results',
                                    use_mock_llm=False):
    """
    使用闭环诊断系统对目标域进行预测

    Args:
        data_dir: 目标域数据目录
        model_path: 模型路径
        output_dir: 输出目录
        use_mock_llm: 是否使用Mock LLM
    """
    print("=== 闭环诊断系统 - 目标域预测 ===")
    print(f"数据目录: {data_dir}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化闭环诊断系统
    print("\n初始化闭环诊断系统...")
    diagnostic_system = ClosedLoopDiagnosticSystem(
        model_path=model_path,
        use_mock_llm=use_mock_llm
    )

    # 获取所有.mat文件
    mat_files = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.mat'):
            mat_files.append(os.path.join(data_dir, filename))

    print(f"发现 {len(mat_files)} 个目标域文件")

    # 处理每个文件
    all_results = []

    for i, file_path in enumerate(mat_files):
        filename = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"处理文件 {i+1}/{len(mat_files)}: {filename}")

        # 加载和预处理
        signal = load_and_preprocess_target_file(file_path)
        if signal is None:
            print(f"跳过文件: {filename}")
            continue

        # 创建信号段
        segments = create_signal_segments(signal, segment_length=2048, overlap_ratio=0.5)
        print(f"信号段数量: {len(segments)}")

        # 限制段数量以节省时间（取前20段）
        if len(segments) > 20:
            segments = segments[:20]
            print(f"限制为前 {len(segments)} 段进行分析")

        # 对每个段进行闭环诊断
        segment_results = []
        for j, segment in enumerate(segments):
            if j % 5 == 0:  # 每5段显示一次进度
                print(f"  处理段 {j+1}/{len(segments)}")

            try:
                segment_id = f"{filename}_segment_{j}"
                result = diagnostic_system.diagnose_single_signal(segment, segment_id)
                segment_results.append(result)
            except Exception as e:
                print(f"  段 {j} 诊断失败: {e}")

        # 聚合段结果
        if segment_results:
            aggregated_result = aggregate_segment_results(segment_results)
            aggregated_result['file_name'] = filename
            aggregated_result['file_path'] = file_path
            all_results.append(aggregated_result)

            print(f"✓ {filename}: {aggregated_result['final_diagnosis']} "
                  f"({aggregated_result['confidence_level']}, "
                  f"投票率: {aggregated_result['vote_ratio']})")
        else:
            print(f"✗ {filename}: 诊断失败")

    # 保存结果
    print(f"\n{'='*50}")
    print("保存诊断结果...")

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细JSON结果
    detailed_json_path = os.path.join(output_dir, f'closed_loop_diagnosis_detailed_{timestamp}.json')
    with open(detailed_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 保存简化CSV结果
    csv_data = []
    for result in all_results:
        csv_data.append({
            'File_Name': result['file_name'],
            'Final_Diagnosis': result['final_diagnosis'],
            'Confidence_Level': result['confidence_level'],
            'Vote_Ratio': result['vote_ratio'],
            'Avg_PATE_Confidence': result['average_pate_confidence'],
            'Avg_Reliability': result['average_reliability'],
            'Segment_Count': result['segment_count'],
            'Representative_Reasoning': result['representative_reasoning'][:200] + '...' if len(result['representative_reasoning']) > 200 else result['representative_reasoning']
        })

    csv_path = os.path.join(output_dir, f'closed_loop_diagnosis_summary_{timestamp}.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # 生成统计报告
    print("\n=== 诊断统计报告 ===")
    fault_distribution = df['Final_Diagnosis'].value_counts()
    print("故障分布:")
    for fault_type, count in fault_distribution.items():
        percentage = count / len(df) * 100
        print(f"  {fault_type}: {count} 个文件 ({percentage:.1f}%)")

    avg_confidence = df['Avg_PATE_Confidence'].mean()
    print(f"\n平均PATE置信度: {avg_confidence:.3f}")

    high_confidence_count = len(df[df['Confidence_Level'] == '高置信度'])
    print(f"高置信度诊断: {high_confidence_count}/{len(df)} ({high_confidence_count/len(df)*100:.1f}%)")

    print(f"\n✓ 闭环诊断完成！")
    print(f"详细结果: {detailed_json_path}")
    print(f"汇总结果: {csv_path}")

    return all_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='闭环诊断系统 - 目标域预测')

    parser.add_argument('--data_dir', type=str, default='data/target_domain',
                       help='目标域数据目录')
    parser.add_argument('--model_path', type=str,
                       default='models_saved/lstm_transfer_all_channels/best_lstm_transfer_full_model.pth',
                       help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--use_mock_llm', action='store_true',
                       help='使用Mock LLM（用于测试）')

    args = parser.parse_args()

    # 运行闭环预测
    results = predict_target_domain_closed_loop(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_mock_llm=args.use_mock_llm
    )

    return results

if __name__ == '__main__':
    main()