"""
模型架构定义
包含Encoder（1D-CNN特征提取器）、ProjectionHead（投影头）、ClassifierHead（分类头）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBlock(nn.Module):
    """1D卷积块：Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pool_size=2):
        super(Conv1DBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class Encoder(nn.Module):
    """
    1D-CNN特征提取器
    输入: (batch_size, 1, signal_length)
    输出: (batch_size, feature_dim)
    """

    def __init__(self, signal_length=2048, feature_dim=512):
        super(Encoder, self).__init__()

        self.signal_length = signal_length
        self.feature_dim = feature_dim

        # 1D-CNN层：逐渐增加通道数，减少序列长度
        self.conv1 = Conv1DBlock(1, 32, kernel_size=7, pool_size=2)      # 2048 -> 1024
        self.conv2 = Conv1DBlock(32, 64, kernel_size=5, pool_size=2)     # 1024 -> 512
        self.conv3 = Conv1DBlock(64, 128, kernel_size=3, pool_size=2)    # 512 -> 256
        self.conv4 = Conv1DBlock(128, 256, kernel_size=3, pool_size=2)   # 256 -> 128

        # 自适应平均池化：将任意长度序列压缩为固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层：映射到特征空间
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 输入形状检查
        if len(x.shape) == 2:  # (batch_size, signal_length)
            x = x.unsqueeze(1)  # 添加通道维度 -> (batch_size, 1, signal_length)

        # 卷积特征提取
        x = self.conv1(x)      # (batch_size, 32, 1024)
        x = self.conv2(x)      # (batch_size, 64, 512)
        x = self.conv3(x)      # (batch_size, 128, 256)
        x = self.conv4(x)      # (batch_size, 256, 128)

        # 全局平均池化
        x = self.adaptive_pool(x)  # (batch_size, 256, 1)
        x = x.squeeze(-1)          # (batch_size, 256)

        # 特征映射
        x = self.dropout(x)
        features = self.fc(x)      # (batch_size, feature_dim)

        return features

class ProjectionHead(nn.Module):
    """
    投影头：用于监督对比学习
    将特征向量映射到低维空间进行对比学习
    """

    def __init__(self, feature_dim=512, projection_dim=128):
        super(ProjectionHead, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, features):
        """
        Args:
            features: (batch_size, feature_dim) 来自Encoder的特征

        Returns:
            projections: (batch_size, projection_dim) 投影后的特征
        """
        projections = self.projection(features)
        # L2归一化：对比学习中的常见做法
        projections = F.normalize(projections, dim=1)
        return projections

class ClassifierHead(nn.Module):
    """
    分类头：用于故障分类
    将特征向量映射到类别概率
    """

    def __init__(self, feature_dim=512, num_classes=4):
        super(ClassifierHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, features):
        """
        Args:
            features: (batch_size, feature_dim) 来自Encoder的特征

        Returns:
            logits: (batch_size, num_classes) 分类logits
        """
        logits = self.classifier(features)
        return logits

class SupConModel(nn.Module):
    """
    监督对比学习模型：Encoder + ProjectionHead
    用于预训练阶段
    """

    def __init__(self, signal_length=2048, feature_dim=512, projection_dim=128):
        super(SupConModel, self).__init__()

        self.encoder = Encoder(signal_length, feature_dim)
        self.projection_head = ProjectionHead(feature_dim, projection_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) 特征向量
            projections: (batch_size, projection_dim) 投影向量
        """
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections

class ClassificationModel(nn.Module):
    """
    分类模型：Encoder + ClassifierHead
    用于微调和最终分类
    """

    def __init__(self, signal_length=2048, feature_dim=512, num_classes=4):
        super(ClassificationModel, self).__init__()

        self.encoder = Encoder(signal_length, feature_dim)
        self.classifier_head = ClassifierHead(feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) 特征向量
            logits: (batch_size, num_classes) 分类logits
        """
        features = self.encoder(x)
        logits = self.classifier_head(features)
        return features, logits

    def get_features(self, x):
        """仅提取特征，不进行分类"""
        return self.encoder(x)

    def classify(self, features):
        """对给定特征进行分类"""
        return self.classifier_head(features)

def create_supcon_model(signal_length=2048, feature_dim=512, projection_dim=128):
    """创建监督对比学习模型"""
    return SupConModel(signal_length, feature_dim, projection_dim)

def create_classification_model(signal_length=2048, feature_dim=512, num_classes=4):
    """创建分类模型"""
    return ClassificationModel(signal_length, feature_dim, num_classes)

class ResBlock1D(nn.Module):
    """
    1D ResNet基础块
    包含两个卷积层和残差连接
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    """
    1D ResNet特征提取器
    适用于振动信号的残差网络架构
    """

    def __init__(self, signal_length=2048, feature_dim=512, layers=[2, 2, 2, 2]):
        super(ResNet1D, self).__init__()

        self.signal_length = signal_length
        self.feature_dim = feature_dim
        self.in_channels = 64

        # 输入层：大卷积核捕获长距离依赖
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet层
        self.layer1 = self._make_layer(64, layers[0], kernel_size=7, stride=1)
        self.layer2 = self._make_layer(128, layers[1], kernel_size=5, stride=2)
        self.layer3 = self._make_layer(256, layers[2], kernel_size=3, stride=2)
        self.layer4 = self._make_layer(512, layers[3], kernel_size=3, stride=2)

        # 全局平均池化和特征映射
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(0.2)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, kernel_size=3, stride=1):
        """构建ResNet层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResBlock1D(self.in_channels, out_channels, kernel_size, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels, kernel_size))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状检查
        if len(x.shape) == 2:  # (batch_size, signal_length)
            x = x.unsqueeze(1)  # 添加通道维度 -> (batch_size, 1, signal_length)

        # 输入处理
        x = self.conv1(x)      # (batch_size, 64, signal_length/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (batch_size, 64, signal_length/4)

        # ResNet层
        x = self.layer1(x)     # (batch_size, 64, signal_length/4)
        x = self.layer2(x)     # (batch_size, 128, signal_length/8)
        x = self.layer3(x)     # (batch_size, 256, signal_length/16)
        x = self.layer4(x)     # (batch_size, 512, signal_length/32)

        # 全局平均池化
        x = self.avgpool(x)    # (batch_size, 512, 1)
        x = x.squeeze(-1)      # (batch_size, 512)

        # 特征映射
        x = self.dropout(x)
        features = self.fc(x)  # (batch_size, feature_dim)

        return features

class ResNetSupConModel(nn.Module):
    """
    基于ResNet1D的监督对比学习模型
    移除投影头，直接使用ResNet特征进行对比学习
    """

    def __init__(self, signal_length=2048, feature_dim=512):
        super(ResNetSupConModel, self).__init__()

        self.encoder = ResNet1D(signal_length, feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) L2归一化的特征向量
        """
        features = self.encoder(x)
        # L2归一化用于对比学习
        features = F.normalize(features, dim=1)
        return features

class ResNetClassificationModel(nn.Module):
    """
    基于ResNet1D的分类模型
    用于迁移学习的分类阶段
    """

    def __init__(self, signal_length=2048, feature_dim=512, num_classes=4):
        super(ResNetClassificationModel, self).__init__()

        self.encoder = ResNet1D(signal_length, feature_dim)
        self.classifier_head = ClassifierHead(feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) 特征向量
            logits: (batch_size, num_classes) 分类logits
        """
        features = self.encoder(x)
        logits = self.classifier_head(features)
        return features, logits

    def get_features(self, x):
        """仅提取特征，不进行分类"""
        return self.encoder(x)

    def classify(self, features):
        """对给定特征进行分类"""
        return self.classifier_head(features)

def create_resnet_supcon_model(signal_length=2048, feature_dim=512):
    """创建基于ResNet1D的监督对比学习模型"""
    return ResNetSupConModel(signal_length, feature_dim)

def create_resnet_classification_model(signal_length=2048, feature_dim=512, num_classes=4):
    """创建基于ResNet1D的分类模型"""
    return ResNetClassificationModel(signal_length, feature_dim, num_classes)

class LSTMEncoder(nn.Module):
    """
    LSTM特征提取器
    适用于时序振动信号的长短期记忆网络
    """

    def __init__(self, signal_length=2048, feature_dim=512, hidden_dim=256, num_layers=3):
        super(LSTMEncoder, self).__init__()

        self.signal_length = signal_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影层：将1D信号投影到LSTM输入维度
        self.input_projection = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制：对LSTM输出进行加权聚合
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # 特征映射层
        self.feature_mapper = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) 特征向量
        """
        batch_size = x.size(0)

        # 输入形状处理
        if len(x.shape) == 3:  # (batch_size, 1, signal_length)
            x = x.squeeze(1)  # (batch_size, signal_length)

        # 转换为序列格式：(batch_size, seq_len, input_size)
        x = x.unsqueeze(-1)  # (batch_size, signal_length, 1)

        # 输入投影
        x = self.input_projection(x)  # (batch_size, signal_length, 64)

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim*2)

        # 注意力加权聚合
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim*2)

        # 特征映射
        features = self.feature_mapper(attended_features)  # (batch_size, feature_dim)

        return features

class LSTMSupConModel(nn.Module):
    """
    基于LSTM的监督对比学习模型
    使用LSTM提取序列特征进行对比学习
    """

    def __init__(self, signal_length=2048, feature_dim=512, hidden_dim=256, num_layers=3):
        super(LSTMSupConModel, self).__init__()

        self.encoder = LSTMEncoder(signal_length, feature_dim, hidden_dim, num_layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) L2归一化的特征向量
        """
        features = self.encoder(x)
        # L2归一化用于对比学习
        features = F.normalize(features, dim=1)
        return features

class LSTMClassificationModel(nn.Module):
    """
    基于LSTM的分类模型
    用于迁移学习的分类阶段
    """

    def __init__(self, signal_length=2048, feature_dim=512, hidden_dim=256, num_layers=3, num_classes=4):
        super(LSTMClassificationModel, self).__init__()

        self.encoder = LSTMEncoder(signal_length, feature_dim, hidden_dim, num_layers)
        self.classifier_head = ClassifierHead(feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, signal_length) 或 (batch_size, 1, signal_length)

        Returns:
            features: (batch_size, feature_dim) 特征向量
            logits: (batch_size, num_classes) 分类logits
        """
        features = self.encoder(x)
        logits = self.classifier_head(features)
        return features, logits

    def get_features(self, x):
        """仅提取特征，不进行分类"""
        return self.encoder(x)

    def classify(self, features):
        """对给定特征进行分类"""
        return self.classifier_head(features)

def create_lstm_supcon_model(signal_length=2048, feature_dim=512, hidden_dim=256, num_layers=3):
    """创建基于LSTM的监督对比学习模型"""
    return LSTMSupConModel(signal_length, feature_dim, hidden_dim, num_layers)

def create_lstm_classification_model(signal_length=2048, feature_dim=512, hidden_dim=256, num_layers=3, num_classes=4):
    """创建基于LSTM的分类模型"""
    return LSTMClassificationModel(signal_length, feature_dim, hidden_dim, num_layers, num_classes)

def load_pretrained_encoder(model, encoder_path):
    """
    加载预训练的Encoder权重

    Args:
        model: 目标模型（包含encoder属性）
        encoder_path: 预训练Encoder权重路径
    """
    try:
        encoder_state = torch.load(encoder_path, map_location='cpu')
        model.encoder.load_state_dict(encoder_state)
        print(f"✓ 成功加载预训练Encoder: {encoder_path}")
    except Exception as e:
        print(f"✗ 加载预训练Encoder失败: {e}")

def freeze_encoder_layers(model, freeze_layers=None):
    """
    冻结Encoder的指定层

    Args:
        model: 目标模型
        freeze_layers: 要冻结的层名列表，None表示冻结前3层卷积
    """
    if freeze_layers is None:
        freeze_layers = ['conv1', 'conv2', 'conv3']

    for name, param in model.encoder.named_parameters():
        for layer_name in freeze_layers:
            if layer_name in name:
                param.requires_grad = False
                print(f"冻结层: {name}")
                break

def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    return total_params, trainable_params

# 测试函数
def test_models():
    """测试模型定义是否正确"""
    print("测试模型架构...")

    # 创建测试数据
    batch_size = 4
    signal_length = 2048
    test_input = torch.randn(batch_size, signal_length)

    # 测试监督对比学习模型
    print("\n1. 测试监督对比学习模型")
    supcon_model = create_supcon_model()
    features, projections = supcon_model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"特征形状: {features.shape}")
    print(f"投影形状: {projections.shape}")
    count_parameters(supcon_model)

    # 测试分类模型
    print("\n2. 测试分类模型")
    cls_model = create_classification_model()
    features, logits = cls_model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"特征形状: {features.shape}")
    print(f"分类logits形状: {logits.shape}")
    count_parameters(cls_model)

    print("\n✓ 模型架构测试通过")

if __name__ == "__main__":
    test_models()