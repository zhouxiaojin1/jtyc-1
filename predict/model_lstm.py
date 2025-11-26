"""
LSTM（长短期记忆网络）

适用：中长期预测、复杂时间依赖关系、序列建模
关键设置：历史窗口 input_len≈(1008~2016)；隐藏层大小 hidden_size=128-256；层数 num_layers=2-3
优点：对长期依赖建模强、适合捕捉复杂时间模式、训练稳定
缺点：训练速度较慢、需要GPU加速、容易过拟合（需要dropout）
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from tqdm import tqdm
import os
import sys
from pathlib import Path
import pickle

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 配置中文字体
from utils.plot_config import setup_chinese_font, apply_plot_style
setup_chinese_font()


class TimeSeriesDataset(Dataset):
    """
    LSTM时间序列数据集
    """
    def __init__(self, data: np.ndarray, input_len: int, output_len: int,
                 time_features: Optional[np.ndarray] = None,
                 region_ids: Optional[np.ndarray] = None,
                 stride: int = 1):
        """
        Parameters:
        -----------
        data : np.ndarray, shape (n_samples, n_regions)
            时间序列数据
        input_len : int
            输入窗口长度
        output_len : int
            输出窗口长度
        time_features : np.ndarray, optional
            时间特征
        region_ids : np.ndarray, optional
            区域ID
        stride : int
            滑动窗口步长
        """
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.time_features = time_features
        self.region_ids = region_ids
        self.stride = stride

        # 生成样本索引
        self.samples = []
        n_timesteps, n_regions = data.shape

        for t in range(input_len, n_timesteps - output_len + 1, stride):
            for r in range(n_regions):
                self.samples.append((t, r))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, r = self.samples[idx]

        # 输入序列
        x = self.data[t - self.input_len:t, r]

        # 目标序列
        y = self.data[t:t + self.output_len, r]

        sample = {
            'x': torch.FloatTensor(x).unsqueeze(-1),  # (input_len, 1)
            'y': torch.FloatTensor(y),  # (output_len,)
        }

        # 时间特征
        if self.time_features is not None:
            time_feat = self.time_features[t - self.input_len:t + self.output_len]
            sample['time_feat'] = torch.FloatTensor(time_feat)

        # 区域ID
        if self.region_ids is not None:
            sample['region_id'] = torch.LongTensor([r])

        return sample


class LSTMModel(nn.Module):
    """
    LSTM预测模型
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 1008,
                 dropout: float = 0.2, use_attention: bool = False,
                 num_regions: int = 0, region_embed_dim: int = 8):
        """
        Parameters:
        -----------
        input_size : int
            输入特征维度
        hidden_size : int
            隐藏层大小
        num_layers : int
            LSTM层数
        output_size : int
            输出序列长度
        dropout : float
            Dropout比例
        use_attention : bool
            是否使用注意力机制
        num_regions : int
            区域数量（用于区域嵌入）
        region_embed_dim : int
            区域嵌入维度
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_attention = use_attention

        # 区域嵌入层
        self.use_region_embed = num_regions > 0
        if self.use_region_embed:
            self.region_embedding = nn.Embedding(num_regions, region_embed_dim)
            self.region_proj = nn.Linear(region_embed_dim, hidden_size)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # 注意力层
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_size)

        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, region_id=None):
        """
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, seq_len, input_size)
            输入序列
        region_id : torch.Tensor, optional, shape (batch_size,)
            区域ID

        Returns:
        --------
        output : torch.Tensor, shape (batch_size, output_size)
            预测序列
        """
        batch_size = x.size(0)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)

        # 注意力机制
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)

        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 融合区域嵌入
        if self.use_region_embed and region_id is not None:
            region_emb = self.region_embedding(region_id.squeeze(-1))  # (batch, region_embed_dim)
            region_feat = self.region_proj(region_emb)  # (batch, hidden_size)
            last_hidden = last_hidden + region_feat

        # 全连接层输出预测
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)  # (batch, output_size)

        return output


class LSTMForecaster:
    """
    LSTM时间序列预测器
    """
    def __init__(self,
                 input_len: int = 2016,
                 output_len: int = 1008,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 use_attention: bool = False,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 patience: int = 2,
                 device: str = 'auto',
                 scaler_type: str = 'standard'):
        """
        Parameters:
        -----------
        input_len : int
            输入序列长度
        output_len : int
            输出序列长度（预测步数）
        hidden_size : int
            LSTM隐藏层大小
        num_layers : int
            LSTM层数
        dropout : float
            Dropout比例
        use_attention : bool
            是否使用注意力机制
        batch_size : int
            批次大小
        learning_rate : float
            学习率
        epochs : int
            训练轮数
        patience : int
            早停耐心值
        device : str
            使用设备 'auto', 'cuda' 或 'cpu'
        scaler_type : str
            数据归一化方式 'standard' 或 'minmax'
        """
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.scaler_type = scaler_type

        # 设备检测
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                print(f"[OK] 检测到GPU: {torch.cuda.get_device_name(0)}，LSTM将使用GPU加速")
            else:
                print("[INFO] 未检测到GPU，LSTM将使用CPU")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def create_time_features(self, df: pd.DataFrame, time_col: str = None) -> np.ndarray:
        """
        创建时间特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        time_col : str
            时间列名（如果None则使用索引）

        Returns:
        --------
        time_features : np.ndarray
            时间特征数组
        """
        if time_col is None:
            if isinstance(df.index, pd.DatetimeIndex):
                time_index = df.index
            else:
                time_index = pd.to_datetime(df.index)
        else:
            time_index = pd.to_datetime(df[time_col])

        # 提取时间特征
        hour = time_index.hour
        day_of_week = time_index.dayofweek
        day_of_month = time_index.day
        month = time_index.month

        # 周期性编码（sin/cos）
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # 是否周末
        is_weekend = (day_of_week >= 5).astype(float)

        # 组合特征
        time_features = np.stack([
            hour_sin, hour_cos,
            day_sin, day_cos,
            month_sin, month_cos,
            is_weekend
        ], axis=1)

        return time_features

    def prepare_data(self, df: pd.DataFrame, train_ratio: float = 0.9,
                    val_ratio: float = 0.05, stride: int = 144) -> Tuple:
        """
        准备训练数据

        Parameters:
        -----------
        df : pd.DataFrame
            输入数据框
        train_ratio : float
            训练集比例
        val_ratio : float
            验证集比例
        stride : int
            滑动窗口步长

        Returns:
        --------
        tuple : (train_loader, val_loader, test_loader, scaler)
        """
        print("[INFO] 准备数据...")

        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_cols].values

        # 数据归一化（仅用训练集拟合，避免数据泄漏）
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        # 划分训练/验证/测试集（原始数据）
        n_samples = len(data)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        # 确保有足够的数据
        min_required = self.input_len + self.output_len
        if n_samples < min_required * 3:
            print(f"[WARNING] 数据量较少，调整划分比例")
            train_ratio = 0.7
            val_ratio = 0.15
            train_size = int(n_samples * train_ratio)
            val_size = int(n_samples * val_ratio)

        train_raw = data[:train_size]
        val_raw = data[train_size - self.input_len:train_size + val_size]
        test_raw = data[train_size + val_size - self.input_len:]

        # 仅在训练集上拟合归一化，再应用到验证/测试
        self.scaler.fit(train_raw)
        train_data = self.scaler.transform(train_raw)
        val_data = self.scaler.transform(val_raw)
        test_data = self.scaler.transform(test_raw)

        # 确保测试集至少有一个样本的数据
        if len(test_data) < self.input_len + self.output_len:
            print(f"[WARNING] 测试集数据不足，调整划分")
            train_size = int(n_samples * 0.7)
            val_size = int(n_samples * 0.1)
            train_raw = data[:train_size]
            val_raw = data[train_size - self.input_len:train_size + val_size]
            test_raw = data[train_size + val_size - self.input_len:]
            self.scaler.fit(train_raw)
            train_data = self.scaler.transform(train_raw)
            val_data = self.scaler.transform(val_raw)
            test_data = self.scaler.transform(test_raw)

        print(f"[INFO] 训练集大小: {train_data.shape}, 验证集大小: {val_data.shape}, 测试集大小: {test_data.shape}")

        # 创建数据集
        train_dataset = TimeSeriesDataset(train_data, self.input_len, self.output_len, stride=stride)
        val_dataset = TimeSeriesDataset(val_data, self.input_len, self.output_len, stride=max(1, self.output_len // 2))
        test_dataset = TimeSeriesDataset(test_data, self.input_len, self.output_len, stride=max(1, self.output_len // 2))

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"[INFO] 训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}, 测试样本数: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        训练模型

        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        """
        print("[INFO] 开始训练...")

        # 创建模型
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_len,
            dropout=self.dropout,
            use_attention=self.use_attention
        ).to(self.device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [训练]')
            for batch in pbar:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                # 前向传播
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = np.mean(train_losses)

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(self.device)
                    y = batch['y'].to(self.device)

                    y_pred = self.model(x)
                    loss = criterion(y_pred, y)

                    # 检查NaN
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
                    else:
                        print(f"[WARNING] 验证损失为NaN，跳过此批次")

            # 如果验证集为空或所有损失都是NaN，使用训练损失
            if len(val_losses) == 0:
                print(f"[WARNING] 验证集为空或全是NaN，使用训练损失")
                avg_val_loss = avg_train_loss
            else:
                avg_val_loss = np.mean(val_losses)

            # 更新学习率
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # 保存训练历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['learning_rate'].append(current_lr)

            print(f'Epoch {epoch+1}/{self.epochs} - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, 学习率: {current_lr:.6f}')

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f'[OK] 验证损失改善，保存最佳模型')
            else:
                patience_counter += 1
                print(f'[INFO] 验证损失未改善 ({patience_counter}/{self.patience})')

                if patience_counter >= self.patience:
                    print(f'[INFO] 早停触发，停止训练')
                    break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f'[OK] 加载最佳模型（验证损失: {best_val_loss:.4f}）')

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Parameters:
        -----------
        data_loader : DataLoader
            数据加载器

        Returns:
        --------
        tuple : (y_true, y_pred)
        """
        print("[INFO] 开始预测...")

        self.model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='预测'):
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                y_pred = self.model(x)

                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(y_pred.cpu().numpy())

        # 检查是否有预测结果
        if len(y_true_list) == 0:
            print("[WARNING] 没有预测结果，返回空数组")
            return np.array([]), np.array([])

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        return y_true, y_pred

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        评估模型

        Parameters:
        -----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值

        Returns:
        --------
        metrics : dict
            评估指标
        """
        # 检查是否为空
        if len(y_true) == 0 or len(y_pred) == 0:
            print("[WARNING] 预测结果为空，返回空指标")
            return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0}

        # 反归一化
        n_samples, n_steps = y_true.shape

        # 创建完整形状的数组用于反归一化
        y_true_full = np.zeros((n_samples, self.scaler.n_features_in_))
        y_pred_full = np.zeros((n_samples, self.scaler.n_features_in_))

        # 只填充第一列（假设只预测一个区域）
        for i in range(n_samples):
            y_true_full[i, 0] = y_true[i, -1]
            y_pred_full[i, 0] = y_pred[i, -1]

        y_true_inv = self.scaler.inverse_transform(y_true_full)[:, 0]
        y_pred_inv = self.scaler.inverse_transform(y_pred_full)[:, 0]

        # 计算指标
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv) * 100

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        return metrics

    def save_model(self, filepath: str):
        """
        保存模型

        Parameters:
        -----------
        filepath : str
            保存路径
        """
        print(f"[INFO] 保存模型到 {filepath}")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_len': self.input_len,
                'output_len': self.output_len,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'use_attention': self.use_attention,
                'scaler_type': self.scaler_type
            },
            'training_history': self.training_history
        }, filepath)

        print("[OK] 模型保存成功")

    def load_model(self, filepath: str):
        """
        加载模型

        Parameters:
        -----------
        filepath : str
            模型路径
        """
        print(f"[INFO] 从 {filepath} 加载模型")

        checkpoint = torch.load(filepath, map_location=self.device)

        # 恢复配置
        config = checkpoint['config']
        self.input_len = config['input_len']
        self.output_len = config['output_len']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_attention = config['use_attention']
        self.scaler_type = config['scaler_type']

        # 重建模型
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_len,
            dropout=self.dropout,
            use_attention=self.use_attention
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']

        print("[OK] 模型加载成功")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        绘制训练历史

        Parameters:
        -----------
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        axes[0].plot(self.training_history['train_loss'], label='训练损失', linewidth=2)
        axes[0].plot(self.training_history['val_loss'], label='验证损失', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('损失', fontsize=12)
        axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 学习率曲线
        axes[1].plot(self.training_history['learning_rate'], color='green', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('学习率', fontsize=12)
        axes[1].set_title('学习率变化曲线', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 训练历史图保存到 {save_path}")

        plt.show()

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        n_samples: int = 5, save_path: Optional[str] = None):
        """
        绘制预测结果

        Parameters:
        -----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值
        n_samples : int
            显示样本数
        save_path : str, optional
            保存路径
        """
        n_samples = min(n_samples, len(y_true))

        fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for i in range(n_samples):
            axes[i].plot(y_true[i], label='真实值', linewidth=2, alpha=0.7)
            axes[i].plot(y_pred[i], label='预测值', linewidth=2, alpha=0.7)
            axes[i].set_xlabel('时间步', fontsize=11)
            axes[i].set_ylabel('值', fontsize=11)
            axes[i].set_title(f'样本 {i+1} 预测结果', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 预测结果图保存到 {save_path}")

        plt.show()


def main():
    """
    主函数：演示LSTM模型训练流程
    """
    print("=" * 80)
    print("LSTM 时间序列预测模型训练")
    print("=" * 80)

    # 加载配置以支持前端数据集切换
    try:
        from utils.config_loader import load_training_config
        cfg = load_training_config()
        data_path = Path(cfg.get('dataset_path') or (project_root / 'dataset' / 'milano_traffic_nid.csv'))
    except Exception:
        data_path = project_root / 'dataset' / 'milano_traffic_nid.csv'
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)

    # 加载数据
    print(f"\n[INFO] 加载数据: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[OK] 数据加载成功，形状: {df.shape}")

    # 选择部分区域进行演示（可以修改）
    selected_regions = df.columns[:10]  # 选择前10个区域
    df_selected = df[selected_regions]
    print(f"[INFO] 选择区域数: {len(selected_regions)}")

    tp = cfg.get('train_params_by_model', {}).get('LSTM', cfg.get('train_params', {})) if 'cfg' in locals() else {}
    mp = cfg.get('model_params_by_model', {}).get('LSTM', cfg.get('model_params', {})) if 'cfg' in locals() else {}

    context_length = int(tp.get('context_length', 2016))
    prediction_length = int(tp.get('prediction_length', 1008))

    if context_length > 0 and len(df_selected) > context_length:
        df_selected = df_selected.tail(context_length)

    forecaster = LSTMForecaster(
        input_len=context_length,
        output_len=prediction_length,
        hidden_size=int(mp.get('hidden_size', 128)),
        num_layers=int(mp.get('num_layers', 2)),
        dropout=float(mp.get('dropout', 0.2)),
        use_attention=False,
        batch_size=int(mp.get('batch_size', 64)),
        learning_rate=float(mp.get('learning_rate', 0.001)),
        epochs=int(mp.get('epochs', 50)),
        patience=2,
        device='auto',
        scaler_type='standard'
    )

    # 准备数据
    train_loader, val_loader, test_loader = forecaster.prepare_data(
        df_selected,
        train_ratio=float(tp.get('train_ratio', 0.85)),
        val_ratio=0.05,
        stride=72
    )

    # 训练模型
    forecaster.train(train_loader, val_loader)

    # 保存模型
    model_path = output_dir / 'lstm_model.pth'
    forecaster.save_model(str(model_path))

    # 绘制训练历史
    history_plot_path = output_dir / 'lstm_training_history.png'
    forecaster.plot_training_history(save_path=str(history_plot_path))

    # 测试集预测
    y_true, y_pred = forecaster.predict(test_loader)

    # 检查是否有预测结果
    if len(y_true) > 0 and len(y_pred) > 0:
        # 评估
        print("\n" + "=" * 80)
        print("测试集评估结果")
        print("=" * 80)
        metrics = forecaster.evaluate(y_true, y_pred)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # 保存评估结果
        metrics_df = pd.DataFrame([metrics])
        metrics_path = output_dir / 'lstm_test_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n[OK] 评估结果保存到 {metrics_path}")

        # 保存预测结果
        pred_df = pd.DataFrame({
            'y_true': y_true.flatten(),
            'y_pred': y_pred.flatten()
        })
        pred_path = output_dir / 'lstm_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        print(f"[OK] 预测结果保存到 {pred_path}")

        # 绘制预测结果
        pred_plot_path = output_dir / 'lstm_predictions_plot.png'
        forecaster.plot_predictions(y_true, y_pred, n_samples=5, save_path=str(pred_plot_path))
    else:
        print("\n[WARNING] 测试集为空，跳过评估和可视化")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
