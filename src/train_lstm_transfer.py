"""
基于LSTM的迁移学习训练脚本
加载预训练的LSTM encoder，进行目标域分类任务训练
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
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import create_lstm_classification_model, load_pretrained_encoder, count_parameters
from src.datasets import StandardDataset

class LSTMTransferTrainer:
    """基于LSTM的迁移学习训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 设置随机种子
        self.set_seed(args.seed)

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        # 初始化LSTM分类模型
        self.model = create_lstm_classification_model(
            signal_length=args.signal_length,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes
        ).to(self.device)

        print("LSTM分类模型结构:")
        count_parameters(self.model)

        # 加载预训练encoder
        if args.pretrained_encoder_path:
            print(f"加载预训练encoder: {args.pretrained_encoder_path}")
            load_pretrained_encoder(self.model, args.pretrained_encoder_path)

            # 冻结encoder部分层（可选）
            if args.freeze_encoder:
                self._freeze_encoder_layers()

        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # 初始化优化器 - 迁移学习使用更小的学习率
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # TensorBoard记录器
        self.writer = SummaryWriter(args.log_dir)

        # 训练记录
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _freeze_encoder_layers(self):
        """冻结encoder的前几层"""
        print("冻结encoder的前几层...")
        freeze_count = 0
        for name, param in self.model.encoder.named_parameters():
            if 'input_projection' in name or 'lstm.weight_ih_l0' in name:
                param.requires_grad = False
                freeze_count += 1
                print(f"冻结参数: {name}")
        print(f"共冻结 {freeze_count} 个参数组")

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

        # 训练数据集 - 使用轻微数据增强
        train_dataset = StandardDataset(
            data_path=self.args.train_data_path,
            augment=True,
            augment_prob=0.2,  # 轻微数据增强
            noise_factor=0.01
        )

        # 验证数据集
        val_dataset = StandardDataset(
            data_path=self.args.val_data_path,
            augment=False,
            augment_prob=0.0,
            noise_factor=0.0
        )

        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False
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
        total_correct = 0
        total_samples = 0
        num_batches = len(train_loader)

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播
            features, logits = self.model(signals)

            # 标签已经是0-3范围，无需调整

            # 计算损失
            loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

            # 打印训练进度
            if batch_idx % self.args.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Batch [{batch_idx}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {batch_acc:.4f} "
                      f"LR: {current_lr:.6f}")

        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_correct / total_samples
        return epoch_loss, epoch_accuracy

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = len(val_loader)
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                features, logits = self.model(signals)

                # 计算损失
                loss = self.criterion(logits, labels)

                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()

                # 收集预测结果用于详细分析
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_correct / total_samples

        return epoch_loss, epoch_accuracy, all_predictions, all_labels

    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'args': self.args
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(self.args.output_dir, 'latest_lstm_transfer_model.pth')
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_lstm_transfer_model.pth')
            torch.save(checkpoint, best_path)

            # 单独保存完整模型用于后续分析
            model_path = os.path.join(self.args.output_dir, 'best_lstm_transfer_full_model.pth')
            torch.save(self.model.state_dict(), model_path)

            print(f"保存最佳LSTM迁移模型: {best_path}")
            print(f"保存完整模型权重: {model_path}")

    def train(self):
        """主训练循环"""
        print("开始LSTM迁移学习训练...")

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders()

        print(f"LSTM层数: {self.args.num_layers}")
        print(f"隐藏维度: {self.args.hidden_dim}")
        print(f"特征维度: {self.args.feature_dim}")
        print(f"分类数量: {self.args.num_classes}")

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # 训练
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_accuracy, val_predictions, val_labels = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)

            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 计算时间
            epoch_time = time.time() - start_time

            print(f"Epoch [{epoch}/{self.args.epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Train Acc: {train_accuracy:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val Acc: {val_accuracy:.4f} "
                  f"Time: {epoch_time:.1f}s")

            # 保存最佳模型
            is_best = val_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = val_accuracy

            self.save_checkpoint(epoch, val_accuracy, is_best)

            # 早停检查
            if self.args.early_stopping > 0:
                if self._should_early_stop():
                    print(f"触发早停，在第{epoch}个epoch停止训练")
                    break

        print("训练完成!")
        print(f"最佳验证准确率: {self.best_accuracy:.4f}")

        # 最终详细分析
        self._final_analysis(val_predictions, val_labels)

        # 关闭TensorBoard
        self.writer.close()

    def _final_analysis(self, predictions, labels):
        """最终模型性能分析"""
        print("\n=== 最终模型性能分析 ===")

        target_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
        print("\n分类报告:")
        print(classification_report(labels, predictions, target_names=target_names))

    def _should_early_stop(self):
        """检查是否应该早停"""
        if len(self.val_accuracies) < self.args.early_stopping:
            return False

        recent_accuracies = self.val_accuracies[-self.args.early_stopping:]
        return all(acc <= self.best_accuracy for acc in recent_accuracies)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LSTM迁移学习训练')

    # 数据参数
    parser.add_argument('--train_data_path', type=str,
                        default='processed_data_de_only/source_train_de_only.npz',
                        help='训练数据路径')
    parser.add_argument('--val_data_path', type=str,
                        default='processed_data_de_only/source_val_de_only.npz',
                        help='验证数据路径')

    # 模型参数
    parser.add_argument('--signal_length', type=int, default=2048, help='信号长度')
    parser.add_argument('--feature_dim', type=int, default=256, help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM隐藏维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--num_classes', type=int, default=4, help='分类类别数')

    # 迁移学习参数
    parser.add_argument('--pretrained_encoder_path', type=str,
                        default='models_saved/lstm_supcon/best_lstm_encoder.pth',
                        help='预训练encoder路径')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='是否冻结encoder部分层')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=10, help='早停patience')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--print_freq', type=int, default=20, help='打印频率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='models_saved/lstm_transfer',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/lstm_transfer',
                        help='日志保存目录')

    args = parser.parse_args()

    # 打印配置
    print("=" * 50)
    print("LSTM迁移学习训练配置:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)

    # 开始训练
    trainer = LSTMTransferTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()