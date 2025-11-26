"""
N-HiTS（全局深度模型）

适用：长预测窗（1008 步）、多季节、多序列共享；对突变/节假日可配外生变量
关键设置：历史窗口 input_len≈(1008~2016)；seasonalities=[144, 1008]；加入时间特征与区域嵌入
优点：对长周期和多季节建模强、推断快；相较 TFT 参数更少、收敛稳定
缺点：需 GPU 更优
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
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 配置中文字体
from utils.plot_config import setup_chinese_font, apply_plot_style
setup_chinese_font()


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集
    """
    def __init__(self, data: np.ndarray, input_len: int, output_len: int,
                 time_features: Optional[np.ndarray] = None,
                 region_ids: Optional[np.ndarray] = None):
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
        """
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.time_features = time_features
        self.region_ids = region_ids

        # 生成样本索引
        self.samples = []
        n_timesteps, n_regions = data.shape

        for t in range(input_len, n_timesteps - output_len + 1):
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
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y),
        }

        # 时间特征
        if self.time_features is not None:
            time_feat = self.time_features[t - self.input_len:t + self.output_len]
            sample['time_feat'] = torch.FloatTensor(time_feat)

        # 区域ID
        if self.region_ids is not None:
            sample['region_id'] = torch.LongTensor([r])

        return sample


class NHiTSBlock(nn.Module):
    """
    N-HiTS的基本块
    """
    def __init__(self, input_size: int, theta_size: int, horizon: int,
                 n_layers: int = 2, hidden_size: int = 512,
                 pooling_size: int = 1, dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.pooling_size = pooling_size

        # 池化层
        if pooling_size > 1:
            self.pooling = nn.AvgPool1d(kernel_size=pooling_size, stride=pooling_size)
            pooled_size = input_size // pooling_size
        else:
            self.pooling = None
            pooled_size = input_size

        # MLP层
        layers = []
        current_size = pooled_size

        for i in range(n_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, theta_size))

        self.mlp = nn.Sequential(*layers)

        # Basis expansion (使用傅里叶基)
        # 分别为backcast和forecast创建basis
        self.register_buffer('basis_backcast', self._create_basis(self.input_size))
        self.register_buffer('basis_forecast', self._create_basis(self.horizon))

    def _create_basis(self, length):
        """
        创建傅里叶基
        """
        t = torch.arange(length, dtype=torch.float32) / length
        basis_functions = []

        # 添加常数项
        basis_functions.append(torch.ones(length))

        # 添加线性项
        basis_functions.append(t)

        # 添加傅里叶项
        n_harmonics = (self.theta_size - 2) // 2
        for i in range(1, n_harmonics + 1):
            basis_functions.append(torch.sin(2 * np.pi * i * t))
            basis_functions.append(torch.cos(2 * np.pi * i * t))

        basis = torch.stack(basis_functions[:self.theta_size], dim=1)  # (length, theta_size)
        return basis

    def forward(self, x):
        """
        x: (batch, input_size)
        """
        # 池化
        if self.pooling is not None:
            x_pooled = self.pooling(x.unsqueeze(1)).squeeze(1)
        else:
            x_pooled = x

        # MLP
        theta = self.mlp(x_pooled)  # (batch, theta_size)

        # Basis expansion
        backcast = torch.matmul(theta, self.basis_backcast.T)  # (batch, input_size)
        forecast = torch.matmul(theta, self.basis_forecast.T)  # (batch, horizon)

        return backcast, forecast


class NHiTS(nn.Module):
    """
    N-HiTS模型
    """
    def __init__(self, input_len: int, output_len: int,
                 n_blocks: int = 3,
                 n_layers: int = 2,
                 hidden_size: int = 512,
                 theta_sizes: Optional[List[int]] = None,
                 pooling_sizes: Optional[List[int]] = None,
                 n_regions: int = 1,
                 region_embed_dim: int = 8,
                 n_time_features: int = 0,
                 dropout: float = 0.1):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.n_blocks = n_blocks
        self.n_regions = n_regions

        # 默认参数
        if theta_sizes is None:
            theta_sizes = [8, 8, 8]
        if pooling_sizes is None:
            # 多尺度池化
            pooling_sizes = [1, 4, 16]

        # 区域嵌入
        if n_regions > 1:
            self.region_embedding = nn.Embedding(n_regions, region_embed_dim)
        else:
            self.region_embedding = None
            region_embed_dim = 0

        # 时间特征投影
        if n_time_features > 0:
            self.time_projection = nn.Linear(n_time_features, 16)
            time_feat_size = 16
        else:
            self.time_projection = None
            time_feat_size = 0

        # 输入投影 - 将额外特征投影到输入空间
        # 如果有额外特征，需要将它们投影后与原始输入相加
        total_extra_dim = region_embed_dim + time_feat_size
        if total_extra_dim > 0:
            self.input_projection = nn.Linear(total_extra_dim, 1)
        else:
            self.input_projection = None

        # 创建N-HiTS块
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = NHiTSBlock(
                input_size=input_len,
                theta_size=theta_sizes[i] if i < len(theta_sizes) else 8,
                horizon=output_len,
                n_layers=n_layers,
                hidden_size=hidden_size,
                pooling_size=pooling_sizes[i] if i < len(pooling_sizes) else 1,
                dropout=dropout
            )
            self.blocks.append(block)

    def forward(self, x, region_id=None, time_feat=None):
        """
        x: (batch, input_len)
        region_id: (batch, 1)
        time_feat: (batch, input_len + output_len, n_time_features)
        """
        batch_size = x.size(0)

        # 开始时x的形状: (batch, input_len)
        x_proj = x

        # 收集额外特征
        extra_features = []

        # 区域嵌入
        if self.region_embedding is not None and region_id is not None:
            region_embed = self.region_embedding(region_id.squeeze(1))  # (batch, embed_dim)
            region_embed = region_embed.unsqueeze(1).expand(-1, self.input_len, -1)  # (batch, input_len, embed_dim)
            extra_features.append(region_embed)

        # 时间特征
        if self.time_projection is not None and time_feat is not None:
            time_feat_input = time_feat[:, :self.input_len, :]  # (batch, input_len, n_time_features)
            time_proj = self.time_projection(time_feat_input)  # (batch, input_len, 16)
            extra_features.append(time_proj)

        # 如果有额外特征，将它们投影并加到输入上
        if len(extra_features) > 0 and self.input_projection is not None:
            # 拼接所有额外特征: (batch, input_len, total_extra_dim)
            extra_concat = torch.cat(extra_features, dim=-1)
            # 投影到每个时间步: (batch, input_len, total_extra_dim) -> (batch, input_len, 1)
            extra_proj = self.input_projection(extra_concat)
            # 压缩最后一维并加到原始输入: (batch, input_len)
            x_proj = x + extra_proj.squeeze(-1)

        # 通过N-HiTS块
        residual = x_proj
        forecast_sum = 0

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

        return forecast_sum


class NHiTSForecaster:
    """
    N-HiTS预测器
    """
    def __init__(self, input_len: int = 1008,
                 output_len: int = 1008,
                 n_blocks: int = 3,
                 n_layers: int = 2,
                 hidden_size: int = 512,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 50,
                 device: str = 'auto'):
        """
        Parameters:
        -----------
        input_len : int
            输入序列长度
        output_len : int
            输出序列长度
        n_blocks : int
            N-HiTS块数量
        hidden_size : int
            隐藏层大小
        learning_rate : float
            学习率
        batch_size : int
            批次大小
        epochs : int
            训练轮数
        device : str
            设备，'auto'自动检测GPU，'cuda'强制GPU，'cpu'强制CPU
        """
        self.input_len = input_len
        self.output_len = output_len
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # 自动检测GPU并测试兼容性
        use_gpu = False
        if device == 'auto' or device == 'cuda':
            if torch.cuda.is_available():
                try:
                    # 测试CUDA是否真正可用
                    test_tensor = torch.zeros(1, device='cuda')
                    _ = test_tensor + 1
                    del test_tensor
                    torch.cuda.empty_cache()
                    use_gpu = True
                    self.device = 'cuda'
                    print(f"[+] 检测到GPU: {torch.cuda.get_device_name(0)}")
                    print(f"    GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                except RuntimeError as e:
                    if 'no kernel image is available' in str(e):
                        print("[!] GPU检测到但不兼容 (RTX 5060 Ti需要更新的PyTorch)")
                        print("    自动切换到CPU模式")
                        self.device = 'cpu'
                    else:
                        raise
            else:
                self.device = 'cpu'
                print("[!] 未检测到GPU，将使用CPU（训练速度会较慢）")
        else:
            self.device = device

        self.model = None
        self.scaler = StandardScaler()

    def create_time_features(self, df: pd.DataFrame, time_col: str) -> np.ndarray:
        """
        创建时间特征
        """
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        hour = df[time_col].dt.hour.values
        dayofweek = df[time_col].dt.dayofweek.values
        is_weekend = (dayofweek >= 5).astype(float)

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * dayofweek / 7)

        time_features = np.stack([hour_sin, hour_cos, dow_sin, dow_cos, is_weekend], axis=1)
        return time_features

    def fit(self, train_df: pd.DataFrame, time_col: str, value_cols: List[str],
            valid_df: Optional[pd.DataFrame] = None):
        """
        训练模型
        """
        print("=" * 60)
        print("开始训练 N-HiTS 模型")
        print(f"输入长度: {self.input_len}, 输出长度: {self.output_len}")
        print(f"设备: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"[>] GPU加速已启用")
            print(f"    批次大小: {self.batch_size}")
            print(f"    隐藏层大小: {self.hidden_size}")
        else:
            print(f"[>] 使用CPU训练")
            print(f"    批次大小: {self.batch_size}")
            print(f"    隐藏层大小: {self.hidden_size}")
        print("=" * 60)

        # 准备数据
        train_data = train_df[value_cols].values

        # 标准化
        train_data_scaled = self.scaler.fit_transform(train_data)

        # 时间特征
        time_features = self.create_time_features(train_df, time_col)

        # 创建数据集
        train_dataset = TimeSeriesDataset(
            train_data_scaled,
            self.input_len,
            self.output_len,
            time_features=time_features,
            region_ids=np.arange(len(value_cols))
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        # 创建模型
        self.model = NHiTS(
            input_len=self.input_len,
            output_len=self.output_len,
            n_blocks=self.n_blocks,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            n_regions=len(value_cols),
            region_embed_dim=8,
            n_time_features=time_features.shape[1]
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 训练
        print(f"\n开始训练，共 {self.epochs} 轮...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                region_id = batch['region_id'].to(self.device)
                time_feat = batch['time_feat'].to(self.device) if 'time_feat' in batch else None

                optimizer.zero_grad()

                # 前向传播
                y_pred = self.model(x, region_id, time_feat)

                # 损失
                loss = F.mse_loss(y_pred, y)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

                pbar.set_postfix({'loss': train_loss / n_batches})

            avg_train_loss = train_loss / n_batches
            scheduler.step(avg_train_loss)

            print(f"Epoch {epoch+1}: 训练损失 = {avg_train_loss:.4f}")

        print("\n训练完成!")

    def predict(self, test_df: pd.DataFrame, time_col: str,
                value_cols: List[str], steps: Optional[int] = None) -> pd.DataFrame:
        """
        预测
        """
        if steps is None:
            steps = self.output_len

        print(f"\n开始预测 {steps} 步...")

        self.model.eval()

        # 准备数据
        test_data = test_df[value_cols].values
        test_data_scaled = self.scaler.transform(test_data)

        # 时间特征 - 需要包含历史和未来
        time_features = self.create_time_features(test_df, time_col)

        # 生成未来时间特征（如果test_df不够长）
        if len(time_features) < self.input_len + steps:
            # 需要生成未来的时间特征
            last_time = pd.to_datetime(test_df[time_col].iloc[-1])

            # 假设10分钟间隔
            future_times = pd.date_range(
                start=last_time + pd.Timedelta(minutes=10),
                periods=steps,
                freq='10min'
            )

            # 创建未来时间特征
            hour = future_times.hour.values
            dayofweek = future_times.dayofweek.values
            is_weekend = (dayofweek >= 5).astype(float)

            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            dow_sin = np.sin(2 * np.pi * dayofweek / 7)
            dow_cos = np.cos(2 * np.pi * dayofweek / 7)

            future_time_features = np.stack([hour_sin, hour_cos, dow_sin, dow_cos, is_weekend], axis=1)

            # 合并历史和未来时间特征
            time_features_extended = np.vstack([time_features, future_time_features])
        else:
            time_features_extended = time_features

        predictions = []

        # 检查数据长度是否足够
        if len(test_data_scaled) < self.input_len:
            raise ValueError(f"测试数据长度 ({len(test_data_scaled)}) 小于输入长度 ({self.input_len})，请提供更多数据")

        with torch.no_grad():
            for r in tqdm(range(len(value_cols)), desc="预测区域"):
                # 取最后input_len个点作为输入
                x = test_data_scaled[-self.input_len:, r]
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)

                region_id = torch.LongTensor([r]).unsqueeze(0).to(self.device)  # shape: (1, 1)

                # 时间特征（包含历史input_len + 未来output_len）
                # 确保有足够的时间特征
                required_time_len = self.input_len + steps
                if len(time_features_extended) < required_time_len:
                    # 如果时间特征不够，从后往前取尽可能多的，并填充
                    available_len = len(time_features_extended)
                    time_feat = time_features_extended[-available_len:, :]
                    # 如果还不够，需要填充或调整
                    if available_len < required_time_len:
                        print(f"[WARNING] 时间特征不足，需要 {required_time_len}，只有 {available_len}")
                        # 使用最后的时间特征重复填充
                        pad_len = required_time_len - available_len
                        last_feat = time_features_extended[-1:, :]
                        padding = np.repeat(last_feat, pad_len, axis=0)
                        time_feat = np.vstack([time_features_extended, padding])
                else:
                    time_feat = time_features_extended[-required_time_len:, :]

                time_feat_tensor = torch.FloatTensor(time_feat).unsqueeze(0).to(self.device)

                # 调试：检查张量形状
                # print(f"x_tensor shape: {x_tensor.shape}")
                # print(f"region_id shape: {region_id.shape}")
                # print(f"time_feat_tensor shape: {time_feat_tensor.shape}")

                # 预测
                try:
                    y_pred = self.model(x_tensor, region_id, time_feat_tensor)
                except Exception as e:
                    print(f"\n[ERROR] 预测失败 - Region {r}")
                    print(f"x_tensor shape: {x_tensor.shape}")
                    print(f"region_id shape: {region_id.shape}")
                    print(f"time_feat_tensor shape: {time_feat_tensor.shape}")
                    print(f"Error: {e}")
                    raise
                y_pred = y_pred.cpu().numpy()[0]

                predictions.append(y_pred[:steps])

        predictions = np.array(predictions).T  # (steps, n_regions)

        # 反标准化
        predictions = self.scaler.inverse_transform(predictions)

        predictions_df = pd.DataFrame(predictions, columns=value_cols)
        return predictions_df

    def evaluate(self, test_df: pd.DataFrame, predictions_df: pd.DataFrame,
                time_col: str) -> pd.DataFrame:
        """评估"""
        value_cols = [col for col in test_df.columns if col != time_col]

        metrics = []
        for region in value_cols:
            if region in predictions_df.columns:
                max_len = min(len(predictions_df), len(test_df))
                y_true = test_df[region].values[:max_len]
                y_pred = predictions_df[region].values[:max_len]

                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if mask.sum() > 0:
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]

                    mae = mean_absolute_error(y_true_clean, y_pred_clean)
                    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100

                    metrics.append({
                        'region': region,
                        'test_mae': mae,
                        'test_rmse': rmse,
                        'test_mape': mape
                    })

        metrics_df = pd.DataFrame(metrics)

        print("\n" + "=" * 60)
        print("测试集评估结果:")
        print("=" * 60)
        print(f"平均MAE: {metrics_df['test_mae'].mean():.2f}")
        print(f"平均RMSE: {metrics_df['test_rmse'].mean():.2f}")
        print(f"平均MAPE: {metrics_df['test_mape'].mean():.2f}%")

        return metrics_df

    def visualize_predictions(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                             predictions_df: pd.DataFrame, time_col: str,
                             region_names: Optional[List[str]] = None,
                             n_regions: int = 3, save_path: Optional[str] = None):
        """可视化"""
        value_cols = [col for col in train_df.columns if col != time_col]

        if region_names is None:
            region_names = np.random.choice(value_cols, min(n_regions, len(value_cols)), replace=False)

        n_plots = len(region_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5*n_plots))
        if n_plots == 1:
            axes = [axes]

        for idx, region in enumerate(region_names):
            ax = axes[idx]

            train_window = min(self.input_len, len(train_df))
            train_series = train_df[region].iloc[-train_window:]
            train_time = np.arange(len(train_series))

            max_len = min(len(predictions_df), len(test_df))
            test_series = test_df[region].iloc[:max_len]
            test_time = np.arange(len(train_series), len(train_series) + len(test_series))

            pred_series = predictions_df[region].values[:max_len]
            pred_time = test_time

            ax.plot(train_time, train_series.values, label='训练数据', color='blue', alpha=0.7)
            ax.plot(test_time, test_series.values, label='真实值', color='green', linewidth=2)
            ax.plot(pred_time, pred_series, label='预测值', color='red', linestyle='--', linewidth=2)

            ax.set_title(f'区域: {region}', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('交通流量', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(x=len(train_series), color='black', linestyle=':', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n可视化已保存: {save_path}")

        plt.close()

    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_len': self.input_len,
                'output_len': self.output_len,
                'n_blocks': self.n_blocks,
                'hidden_size': self.hidden_size
            }
        }, save_path)
        print(f"\n模型已保存: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("N-HiTS 时间序列预测模型")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        from utils.config_loader import load_training_config
        cfg = load_training_config()
        data_path = cfg.get('dataset_path')
        if data_path is None or not os.path.exists(data_path):
            data_path = os.path.join(script_dir, '..', 'dataset', 'milano_traffic_nid.csv')
    except Exception:
        data_path = os.path.join(script_dir, '..', 'dataset', 'milano_traffic_nid.csv')
    print(f"[INFO] 数据路径: {data_path}")
    df = pd.read_csv(data_path)

    train_params = {}
    model_params = {}
    try:
        from utils.config_loader import load_training_config
        cfg = load_training_config()
        tp_all = cfg.get('train_params_by_model', {})
        mp_all = cfg.get('model_params_by_model', {})
        train_params = tp_all.get('N-HiTS', cfg.get('train_params', {})) or {}
        model_params = mp_all.get('N-HiTS', cfg.get('model_params', {})) or {}
    except Exception:
        train_params = {}
        model_params = {}

    split_ratio = float(train_params.get('train_ratio', 0.9))
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    time_col = df.columns[0]
    all_regions = [col for col in df.columns if col != time_col]
    sample_regions = all_regions[:5]

    train_sample = train_df[[time_col] + sample_regions]
    test_sample = test_df[[time_col] + sample_regions]

    context_length = int(train_params.get('context_length', 1008))
    prediction_length = int(train_params.get('prediction_length', 288))
    n_blocks = int(model_params.get('n_blocks', 3))
    n_layers = int(model_params.get('n_layers', 2))
    hidden_size = int(model_params.get('hidden_size', 512))
    batch_size = int(model_params.get('batch_size', 64))
    epochs = int(model_params.get('epochs', 50))
    learning_rate = float(model_params.get('learning_rate', 0.001))

    if context_length > 0 and len(train_sample) > context_length:
        train_sample = train_sample.tail(context_length).reset_index(drop=True)

    forecaster = NHiTSForecaster(
        input_len=context_length,
        output_len=prediction_length,
        n_blocks=n_blocks,
        n_layers=n_layers,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        device='auto'
    )

    forecaster.fit(train_sample, time_col, sample_regions)

    # 创建output目录（使用绝对路径）
    output_dir = os.path.join(script_dir, '..', 'output')
    output_dir = os.path.abspath(output_dir)  # 转换为绝对路径
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 预测：以训练集末尾为起点，对齐测试集起始段
    predictions = forecaster.predict(train_sample, time_col, sample_regions, steps=prediction_length)
    predictions.to_csv(os.path.join(output_dir, 'nhits_predictions.csv'), index=False)

    # 评估
    metrics = forecaster.evaluate(test_sample, predictions, time_col)
    metrics.to_csv(os.path.join(output_dir, 'nhits_test_metrics.csv'), index=False)

    # 可视化
    forecaster.visualize_predictions(
        train_sample, test_sample, predictions, time_col,
        region_names=sample_regions[:3],
        save_path=os.path.join(output_dir, 'nhits_predictions_plot.png')
    )

    # 保存
    forecaster.save_model(os.path.join(output_dir, 'nhits_model.pth'))

    print("\nN-HiTS model completed!")
    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    main()
