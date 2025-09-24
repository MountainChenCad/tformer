"""
基于LSTM的监督对比学习训练脚本
使用LSTM+注意力机制提取时序特征，进行有监督对比学习
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import create_lstm_supcon_model, count_parameters
from src.datasets import SupConDataset, supcon_collate_fn
from src.losses import SupConLoss

class LSTMSupConTrainer:
    """基于LSTM的监督对比学习训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 设置随机种子
        self.set_seed(args.seed)

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        # 初始化LSTM模型
        self.model = create_lstm_supcon_model(
            signal_length=args.signal_length,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        ).to(self.device)

        print("LSTM模型结构:")
        count_parameters(self.model)

        # 初始化损失函数
        self.criterion = SupConLoss(temperature=args.temperature).to(self.device)

        # 初始化优化器 - LSTM通常需要更小的学习率
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 学习率调度器 - 针对LSTM优化
        self.scheduler = self._create_scheduler()

        # TensorBoard记录器
        self.writer = SummaryWriter(args.log_dir)

        # 训练记录
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def _create_scheduler(self):
        """创建适合LSTM的学习率调度器"""
        # LSTM训练通常需要更稳定的学习率调度
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=5,
            min_lr=1e-6
        )

    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def create_dataloaders(self):
        """创建数据加载器"""
        print("创建数据加载器...")

        # 训练数据集 - LSTM对噪声更敏感，降低增强强度
        train_dataset = SupConDataset(
            data_path=self.args.train_data_path,
            augment_prob=0.3,  # 进一步降低增强概率
            noise_factor=0.02  # 降低噪声强度
        )

        # 验证数据集
        val_dataset = SupConDataset(
            data_path=self.args.val_data_path,
            augment_prob=0.0,
            noise_factor=0.0
        )

        # 数据加载器 - LSTM序列处理，可以用较小的batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            collate_fn=supcon_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=supcon_collate_fn
        )

        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(val_dataset)}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")

        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (view1, view2, labels) in enumerate(train_loader):
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播
            features1 = self.model(view1)  # (batch_size, feature_dim)
            features2 = self.model(view2)  # (batch_size, feature_dim)

            # 堆叠两个视图的特征用于对比学习
            features = torch.stack([features1, features2], dim=1)  # (batch_size, 2, feature_dim)

            # 计算对比损失
            loss = self.criterion(features, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 - LSTM训练中很重要
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # 打印训练进度
            if batch_idx % self.args.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Batch [{batch_idx}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {current_lr:.6f}")

        return total_loss / num_batches

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for view1, view2, labels in val_loader:
                view1 = view1.to(self.device, non_blocking=True)
                view2 = view2.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                features1 = self.model(view1)
                features2 = self.model(view2)

                # 堆叠特征
                features = torch.stack([features1, features2], dim=1)

                # 计算损失
                loss = self.criterion(features, labels)
                total_loss += loss.item()

        return total_loss / num_batches

    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(self.args.output_dir, 'latest_lstm_model.pth')
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_lstm_model.pth')
            torch.save(checkpoint, best_path)

            # 单独保存encoder用于迁移学习
            encoder_path = os.path.join(self.args.output_dir, 'best_lstm_encoder.pth')
            torch.save(self.model.encoder.state_dict(), encoder_path)

            print(f"保存最佳LSTM模型: {best_path}")
            print(f"保存最佳LSTM encoder: {encoder_path}")

    def train(self):
        """主训练循环"""
        print("开始LSTM监督对比学习训练...")

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders()

        print(f"LSTM层数: {self.args.num_layers}")
        print(f"隐藏维度: {self.args.hidden_dim}")
        print(f"温度参数: {self.args.temperature}")

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss = self.validate(val_loader)

            # 更新学习率 - ReduceLROnPlateau基于验证损失
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 计算时间
            epoch_time = time.time() - start_time

            print(f"Epoch [{epoch}/{self.args.epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Time: {epoch_time:.1f}s")

            # 保存最佳模型
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

            # 早停检查
            if self.args.early_stopping > 0:
                if self._should_early_stop():
                    print(f"触发早停，在第{epoch}个epoch停止训练")
                    break

        print("训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")

        # 关闭TensorBoard
        self.writer.close()

    def _should_early_stop(self):
        """检查是否应该早停"""
        if len(self.val_losses) < self.args.early_stopping:
            return False

        recent_losses = self.val_losses[-self.args.early_stopping:]
        return all(loss >= self.best_loss for loss in recent_losses)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LSTM监督对比学习训练')

    # 数据参数
    parser.add_argument('--train_data_path', type=str, default='processed_data_de_only/source_train_de_only.npz',
                        help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default='processed_data_de_only/source_val_de_only.npz',
                        help='验证数据路径')

    # 模型参数
    parser.add_argument('--signal_length', type=int, default=2048, help='信号长度')
    parser.add_argument('--feature_dim', type=int, default=256, help='特征维度') # 降低到256维
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM隐藏维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--temperature', type=float, default=0.07, help='对比学习温度参数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小') # LSTM用较小batch size
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=15, help='早停patience')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--print_freq', type=int, default=50, help='打印频率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='models_saved/lstm_supcon',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/lstm_supcon',
                        help='日志保存目录')

    args = parser.parse_args()

    # 打印配置
    print("=" * 50)
    print("LSTM监督对比学习训练配置:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)

    # 开始训练
    trainer = LSTMSupConTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()