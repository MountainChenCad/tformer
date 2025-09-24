"""
数据集类定义
包含SupConDataset（监督对比学习）和StandardDataset（标准分类）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class BaseDataset(Dataset):
    """基础数据集类"""

    def __init__(self, data_path):
        """
        Args:
            data_path: .npz文件路径
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 加载数据
        data = np.load(data_path)
        self.samples = data['samples']

        # 检查是否有标签（源域数据有标签，目标域数据无标签）
        if 'labels' in data:
            self.labels = data['labels']
            self.has_labels = True
        else:
            self.labels = None
            self.has_labels = False

        # 检查是否有通道信息（多通道数据）
        if 'channels' in data:
            self.channels = data['channels']
            self.has_channels = True
        else:
            self.channels = None
            self.has_channels = False

        print(f"加载数据: {data_path}")
        print(f"样本数量: {len(self.samples)}")
        if self.has_labels:
            print(f"标签分布: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.samples)

class SupConDataset(BaseDataset):
    """
    监督对比学习数据集
    为每个样本生成两个增强视图，用于对比学习
    """

    def __init__(self, data_path, augment_prob=0.8, noise_factor=0.1):
        """
        Args:
            data_path: .npz文件路径
            augment_prob: 数据增强概率
            noise_factor: 噪声强度
        """
        super().__init__(data_path)

        if not self.has_labels:
            raise ValueError("监督对比学习需要标签数据")

        self.augment_prob = augment_prob
        self.noise_factor = noise_factor

    def augment_signal(self, signal):
        """
        对信号进行数据增强

        Args:
            signal: 输入信号 (numpy array)

        Returns:
            augmented_signal: 增强后的信号
        """
        augmented = signal.copy()

        # 随机应用增强策略
        if random.random() < self.augment_prob:
            # 1. 添加高斯噪声
            if random.random() < 0.5:
                noise = np.random.normal(0, self.noise_factor, signal.shape)
                augmented = augmented + noise

            # 2. 幅值缩放
            if random.random() < 0.5:
                scale_factor = random.uniform(0.8, 1.2)
                augmented = augmented * scale_factor

            # 3. 时间偏移（循环移位）
            if random.random() < 0.5:
                shift = random.randint(-len(signal)//10, len(signal)//10)
                augmented = np.roll(augmented, shift)

            # 4. 随机遮蔽
            if random.random() < 0.3:
                mask_length = random.randint(1, len(signal)//20)
                mask_start = random.randint(0, len(signal) - mask_length)
                augmented[mask_start:mask_start + mask_length] = 0

        return augmented

    def __getitem__(self, idx):
        """
        返回一个样本的两个增强视图和标签

        Returns:
            view1: 第一个增强视图
            view2: 第二个增强视图
            label: 标签
        """
        sample = self.samples[idx]
        label = self.labels[idx]

        # 生成两个不同的增强视图
        view1 = self.augment_signal(sample)
        view2 = self.augment_signal(sample)

        # 转换为tensor
        view1 = torch.FloatTensor(view1)
        view2 = torch.FloatTensor(view2)
        label = torch.LongTensor([label])[0]

        return view1, view2, label

class StandardDataset(BaseDataset):
    """
    标准数据集
    用于分类训练、验证和测试
    """

    def __init__(self, data_path, augment=False, augment_prob=0.5, noise_factor=0.05):
        """
        Args:
            data_path: .npz文件路径
            augment: 是否进行数据增强
            augment_prob: 数据增强概率
            noise_factor: 噪声强度
        """
        super().__init__(data_path)

        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_factor = noise_factor

    def augment_signal(self, signal):
        """轻度数据增强，用于分类训练"""
        if not self.augment or random.random() > self.augment_prob:
            return signal

        augmented = signal.copy()

        # 轻度噪声
        if random.random() < 0.6:
            noise = np.random.normal(0, self.noise_factor, signal.shape)
            augmented = augmented + noise

        # 轻度缩放
        if random.random() < 0.4:
            scale_factor = random.uniform(0.95, 1.05)
            augmented = augmented * scale_factor

        return augmented

    def __getitem__(self, idx):
        """
        返回样本和标签（如果有）

        Returns:
            sample: 信号样本
            label: 标签（如果有的话）
        """
        sample = self.samples[idx]

        # 数据增强
        if self.augment:
            sample = self.augment_signal(sample)

        # 转换为tensor
        sample = torch.FloatTensor(sample)

        if self.has_labels:
            label = torch.LongTensor([self.labels[idx]])[0]
            return sample, label
        else:
            # 目标域数据没有标签，返回样本和索引
            return sample, idx

class TargetDataset(BaseDataset):
    """
    目标域数据集
    专门处理目标域的无标签数据，支持文件名映射
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: .npz文件路径（target_data.npz）
        """
        super().__init__(data_path)

        # 加载目标域特有的信息
        data = np.load(data_path)
        self.filenames = data['filenames']  # 每个样本对应的文件名

        if 'file_sample_counts' in data:
            self.file_sample_counts = data['file_sample_counts']
        else:
            self.file_sample_counts = None

        print(f"目标域文件数: {len(set(self.filenames))}")

    def __getitem__(self, idx):
        """
        返回样本、索引和文件名

        Returns:
            sample: 信号样本
            idx: 样本索引
            filename: 对应的文件名
        """
        sample = self.samples[idx]
        filename = self.filenames[idx]

        # 转换为tensor
        sample = torch.FloatTensor(sample)

        return sample, idx, filename

    def get_file_indices(self, filename):
        """获取指定文件的所有样本索引"""
        indices = []
        for i, fname in enumerate(self.filenames):
            if fname == filename:
                indices.append(i)
        return indices

    def get_unique_files(self):
        """获取所有唯一的文件名"""
        return list(set(self.filenames))

def create_dataloaders(train_data_path, val_data_path, batch_size=64, num_workers=4, dataset_type='standard'):
    """
    创建训练和验证数据加载器

    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        dataset_type: 数据集类型 ('supcon' 或 'standard')

    Returns:
        train_loader, val_loader: 数据加载器
    """
    from torch.utils.data import DataLoader

    if dataset_type == 'supcon':
        train_dataset = SupConDataset(train_data_path)
        val_dataset = SupConDataset(val_data_path, augment_prob=0.0)  # 验证时不增强
    else:
        train_dataset = StandardDataset(train_data_path, augment=True)
        val_dataset = StandardDataset(val_data_path, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 对比学习需要保持批次大小一致
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def create_target_dataloader(target_data_path, batch_size=64, num_workers=4):
    """
    创建目标域数据加载器

    Args:
        target_data_path: 目标域数据路径
        batch_size: 批次大小
        num_workers: 工作进程数

    Returns:
        target_loader: 目标域数据加载器
    """
    from torch.utils.data import DataLoader

    target_dataset = TargetDataset(target_data_path)

    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,  # 目标域推理时不打乱
        num_workers=num_workers,
        pin_memory=True
    )

    return target_loader

# 自定义collate函数（用于监督对比学习）
def supcon_collate_fn(batch):
    """
    监督对比学习的批次整理函数
    保持两个视图分离以便于损失函数处理
    """
    view1_list, view2_list, label_list = zip(*batch)

    # 保持两个视图分离
    view1 = torch.stack(view1_list, dim=0)
    view2 = torch.stack(view2_list, dim=0)
    labels = torch.stack(label_list, dim=0)

    return view1, view2, labels

# 测试函数
def test_datasets():
    """测试数据集类"""
    print("测试数据集类...")

    # 创建模拟数据
    test_data_path = 'test_data.npz'
    samples = np.random.randn(100, 2048)
    labels = np.random.randint(0, 4, 100)

    np.savez(test_data_path, samples=samples, labels=labels)

    try:
        # 测试监督对比学习数据集
        print("\n1. 测试监督对比学习数据集")
        supcon_dataset = SupConDataset(test_data_path)
        view1, view2, label = supcon_dataset[0]
        print(f"View1形状: {view1.shape}")
        print(f"View2形状: {view2.shape}")
        print(f"标签: {label}")

        # 测试标准数据集
        print("\n2. 测试标准数据集")
        std_dataset = StandardDataset(test_data_path, augment=True)
        sample, label = std_dataset[0]
        print(f"样本形状: {sample.shape}")
        print(f"标签: {label}")

        # 测试数据加载器
        print("\n3. 测试数据加载器")
        train_loader, val_loader = create_dataloaders(
            test_data_path, test_data_path,
            batch_size=8, num_workers=0,
            dataset_type='standard'
        )

        for batch_idx, (samples, labels) in enumerate(train_loader):
            print(f"批次 {batch_idx}: 样本形状 {samples.shape}, 标签形状 {labels.shape}")
            if batch_idx >= 2:  # 只测试几个批次
                break

        print("✓ 数据集测试通过")

    finally:
        # 清理测试文件
        if os.path.exists(test_data_path):
            os.remove(test_data_path)

if __name__ == "__main__":
    test_datasets()