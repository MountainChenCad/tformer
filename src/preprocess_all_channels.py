"""
基于所有通道(DE, FE, BA)的数据预处理脚本
扩增源域数据集，用于对比学习训练
"""

import os
import glob
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def extract_fault_type_from_filename(filename):
    """从文件名提取故障类型"""
    base = os.path.basename(filename).lower()

    # 正常状态
    if 'normal' in base or 'n_' in base:
        return 0, 'Normal'

    # 内圈故障
    if base.startswith('ir') or 'inner' in base:
        return 1, 'Inner'

    # 外圈故障
    if base.startswith('or') or 'outer' in base:
        return 2, 'Outer'

    # 滚动体故障
    if base.startswith('b') and ('007' in base or '014' in base or '021' in base or '028' in base):
        return 3, 'Ball'

    return -1, 'Unknown'

def load_matlab_file(filepath):
    """加载MATLAB文件并提取所有通道数据"""
    try:
        mat_data = sio.loadmat(filepath)

        # 获取所有变量名
        var_names = [k for k in mat_data.keys() if not k.startswith('__')]

        # 寻找DE, FE, BA数据
        de_data = None
        fe_data = None
        ba_data = None

        for var_name in var_names:
            var_lower = var_name.lower()
            if 'de_time' in var_lower or (var_name.endswith('DE_time')):
                de_data = mat_data[var_name].flatten()
            elif 'fe_time' in var_lower or (var_name.endswith('FE_time')):
                fe_data = mat_data[var_name].flatten()
            elif 'ba_time' in var_lower or (var_name.endswith('BA_time')):
                ba_data = mat_data[var_name].flatten()

        return de_data, fe_data, ba_data

    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None, None, None

def segment_signal(signal, segment_length=2048, overlap_ratio=0.5):
    """将信号分段，支持50%重叠"""
    if signal is None or len(signal) < segment_length:
        return []

    stride = int(segment_length * (1 - overlap_ratio))
    segments = []

    for i in range(0, len(signal) - segment_length + 1, stride):
        segment = signal[i:i + segment_length]
        segments.append(segment)

    return segments

def process_all_channels_data(data_dir, output_dir, segment_length=2048):
    """处理所有通道数据，创建扩增数据集"""
    print(f"Processing data from: {data_dir}")

    # 寻找所有.mat文件
    mat_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root, file))

    print(f"Found {len(mat_files)} .mat files")

    all_samples = []
    all_labels = []
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    file_info = []

    for mat_file in mat_files:
        # 提取故障类型
        label, fault_name = extract_fault_type_from_filename(mat_file)
        if label == -1:
            print(f"Skipping unknown file: {mat_file}")
            continue

        # 加载数据
        de_data, fe_data, ba_data = load_matlab_file(mat_file)

        # 处理每个通道
        channels = [('DE', de_data), ('FE', fe_data), ('BA', ba_data)]

        for channel_name, channel_data in channels:
            if channel_data is None:
                continue

            # 分段处理
            segments = segment_signal(channel_data, segment_length)

            if len(segments) > 0:
                print(f"  {os.path.basename(mat_file)} - {channel_name}: {len(segments)} segments, Label: {fault_name}")

                for seg_idx, segment in enumerate(segments):
                    # 质量检查：去除全零或异常片段
                    if np.std(segment) < 1e-6 or np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
                        continue

                    # 鲁棒标准化
                    scaler = RobustScaler()
                    normalized_segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()

                    all_samples.append(normalized_segment)
                    all_labels.append(label)
                    label_counts[label] += 1

                    file_info.append({
                        'file': os.path.basename(mat_file),
                        'channel': channel_name,
                        'segment': seg_idx,
                        'label': label,
                        'fault_name': fault_name
                    })

    # 转换为numpy数组
    samples_array = np.array(all_samples)
    labels_array = np.array(all_labels)

    # 数据集统计
    print(f"\n=== 数据集统计 ===")
    print(f"总样本数: {len(samples_array)}")
    print(f"信号长度: {segment_length}")
    print(f"标签分布:")
    fault_names = ['Normal', 'Inner', 'Outer', 'Ball']
    for i, name in enumerate(fault_names):
        print(f"  {name}: {label_counts[i]} 样本 ({label_counts[i]/len(samples_array)*100:.1f}%)")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据分割：80%训练，20%验证
    from sklearn.model_selection import train_test_split

    train_data, val_data, train_labels, val_labels = train_test_split(
        samples_array, labels_array,
        test_size=0.2,
        stratify=labels_array,
        random_state=42
    )

    # 保存训练集
    train_path = os.path.join(output_dir, 'source_train_all_channels.npz')
    np.savez_compressed(train_path, samples=train_data, labels=train_labels)
    print(f"训练集保存到: {train_path}")
    print(f"训练样本: {len(train_data)}")

    # 保存验证集
    val_path = os.path.join(output_dir, 'source_val_all_channels.npz')
    np.savez_compressed(val_path, samples=val_data, labels=val_labels)
    print(f"验证集保存到: {val_path}")
    print(f"验证样本: {len(val_data)}")

    # 保存完整数据集
    all_path = os.path.join(output_dir, 'source_all_all_channels.npz')
    np.savez_compressed(all_path, samples=samples_array, labels=labels_array)
    print(f"完整数据集保存到: {all_path}")

    # 保存处理信息
    import json
    info_dict = {
        'total_samples': len(samples_array),
        'segment_length': segment_length,
        'overlap_ratio': 0.5,
        'label_distribution': label_counts,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'channels_used': ['DE', 'FE', 'BA'],
        'normalization': 'RobustScaler',
        'file_count': len(mat_files),
        'fault_names': fault_names
    }

    info_path = os.path.join(output_dir, 'processing_info_all_channels.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)
    print(f"处理信息保存到: {info_path}")

    return train_path, val_path, all_path

def main():
    # 数据路径配置
    source_data_dir = '../data/源域数据集'
    output_dir = '../processed_data_all_channels'

    print("=== 基于所有通道的数据预处理 ===")
    print("处理DE, FE, BA三个通道数据，扩增源域数据集")

    # 检查数据目录
    if not os.path.exists(source_data_dir):
        print(f"Error: 源数据目录不存在: {source_data_dir}")
        print("请确保已下载并解压源域数据集")
        return

    # 处理数据
    try:
        train_path, val_path, all_path = process_all_channels_data(
            source_data_dir, output_dir
        )

        print(f"\n✅ 所有通道数据预处理完成!")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 可用于监督对比学习训练")

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()