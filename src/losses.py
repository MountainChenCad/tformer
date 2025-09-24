"""
损失函数定义
包含监督对比损失(Supervised Contrastive Loss)和其他相关损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)

    论文: Supervised Contrastive Learning
    https://arxiv.org/abs/2004.11362

    该损失函数旨在将相同类别的样本在特征空间中拉近，
    将不同类别的样本推远。
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        Args:
            temperature: 温度参数，控制软化程度
            contrast_mode: 对比模式，'all' 或 'one'
            base_temperature: 基础温度参数
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        计算监督对比损失

        Args:
            features: (batch_size, projection_dim) 投影后的特征向量
            labels: (batch_size,) 标签，如果为None则计算自监督对比损失
            mask: (batch_size, batch_size) 对比掩码，如果为None则根据labels自动生成

        Returns:
            loss: 监督对比损失值
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('features需要是3维张量: (batch_size, n_views, projection_dim)')

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('不能同时指定labels和mask')

        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('标签数量与批次大小不匹配')

            # 创建标签掩码：相同标签的位置为True
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 视图数量
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('未知的对比模式: {}'.format(self.contrast_mode))

        # 计算锚点特征与对比特征的相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 构建掩码
        mask = mask.repeat(anchor_count, contrast_count)

        # 移除自己与自己的对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算平均log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 损失值
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数
    常用于自监督学习中的对比学习
    """

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: (2*batch_size, projection_dim) 两个视图的特征拼接

        Returns:
            loss: InfoNCE损失值
        """
        batch_size = features.shape[0] // 2

        # 分离两个视图
        z1 = features[:batch_size]
        z2 = features[batch_size:]

        # 计算所有对的相似度
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建正样本对的掩码
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)

        # 移除自己与自己的相似度
        sim_matrix = sim_matrix - torch.eye(2 * batch_size).to(features.device) * 1e9

        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (pos_mask.to(features.device) * exp_sim).sum(dim=1)
        total_sim = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sim / total_sim).mean()

        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) 预测logits
            targets: (batch_size,) 真实标签

        Returns:
            loss: Focal Loss值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失
    减少模型过拟合，提高泛化能力
    """

    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) 预测logits
            targets: (batch_size,) 真实标签

        Returns:
            loss: 标签平滑损失值
        """
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        smooth_targets = targets_one_hot * (1 - self.smoothing) + \
                        self.smoothing / self.num_classes

        loss = (-smooth_targets * log_probs).sum(dim=1).mean()

        return loss

class CenterLoss(nn.Module):
    """
    Center Loss
    用于增强特征的类内聚合度
    """

    def __init__(self, num_classes, feature_dim, lambda_c=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c

        # 初始化类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, feature_dim) 特征向量
            labels: (batch_size,) 真实标签

        Returns:
            loss: Center Loss值
        """
        batch_size = features.shape[0]

        # 获取对应标签的中心
        centers_batch = self.centers[labels]

        # 计算特征与中心的距离
        center_loss = F.mse_loss(features, centers_batch)

        return self.lambda_c * center_loss

class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合多种损失函数，用于迁移学习微调阶段
    """

    def __init__(self,
                 num_classes=4,
                 feature_dim=512,
                 use_focal=True,
                 use_center=True,
                 use_label_smoothing=False,
                 alpha_focal=1.0,
                 gamma_focal=2.0,
                 lambda_center=0.1,
                 smoothing=0.1):
        super(CombinedLoss, self).__init__()

        self.use_focal = use_focal
        self.use_center = use_center
        self.use_label_smoothing = use_label_smoothing

        # 基础交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()

        # Focal Loss
        if use_focal:
            self.focal_loss = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)

        # Center Loss
        if use_center:
            self.center_loss = CenterLoss(num_classes, feature_dim, lambda_center)

        # 标签平滑损失
        if use_label_smoothing:
            self.label_smooth_loss = LabelSmoothingLoss(num_classes, smoothing)

    def forward(self, logits, features, labels):
        """
        Args:
            logits: (batch_size, num_classes) 分类logits
            features: (batch_size, feature_dim) 特征向量
            labels: (batch_size,) 真实标签

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        total_loss = 0

        # 基础分类损失
        if self.use_focal:
            classification_loss = self.focal_loss(logits, labels)
            loss_dict['focal_loss'] = classification_loss.item()
        elif self.use_label_smoothing:
            classification_loss = self.label_smooth_loss(logits, labels)
            loss_dict['label_smooth_loss'] = classification_loss.item()
        else:
            classification_loss = self.ce_loss(logits, labels)
            loss_dict['ce_loss'] = classification_loss.item()

        total_loss += classification_loss

        # Center Loss
        if self.use_center:
            center_loss = self.center_loss(features, labels)
            total_loss += center_loss
            loss_dict['center_loss'] = center_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

# 测试函数
def test_losses():
    """测试损失函数"""
    print("测试损失函数...")

    batch_size = 8
    num_classes = 4
    projection_dim = 128
    feature_dim = 512

    # 创建测试数据
    features = torch.randn(batch_size, 2, projection_dim)  # 两个视图
    labels = torch.randint(0, num_classes, (batch_size,))
    logits = torch.randn(batch_size, num_classes)
    feature_vectors = torch.randn(batch_size, feature_dim)

    # 测试监督对比损失
    print("\n1. 测试监督对比损失")
    supcon_loss = SupConLoss()
    loss_value = supcon_loss(features, labels)
    print(f"SupCon Loss: {loss_value.item():.4f}")

    # 测试Focal Loss
    print("\n2. 测试Focal Loss")
    focal_loss = FocalLoss()
    loss_value = focal_loss(logits, labels)
    print(f"Focal Loss: {loss_value.item():.4f}")

    # 测试Center Loss
    print("\n3. 测试Center Loss")
    center_loss = CenterLoss(num_classes, feature_dim)
    loss_value = center_loss(feature_vectors, labels)
    print(f"Center Loss: {loss_value.item():.4f}")

    # 测试组合损失
    print("\n4. 测试组合损失")
    combined_loss = CombinedLoss(num_classes, feature_dim)
    total_loss, loss_dict = combined_loss(logits, feature_vectors, labels)
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")

    print("\n✓ 损失函数测试通过")

if __name__ == "__main__":
    test_losses()