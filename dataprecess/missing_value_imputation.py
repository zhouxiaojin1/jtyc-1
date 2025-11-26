"""
缺失值填补模块 - STL + 状态空间卡尔曼平滑器（Seasonal Kalman Smoothing）

原理：
1. 先使用STL分解提取季节成分（日季节性=144、周季节性=1008）
2. 在状态空间模型中用卡尔曼滤波/平滑对趋势与残差进行插补

适配性：能同时处理长短季节与突发缺口，插补平滑而不削弱季节结构
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class KalmanSmoother:
    """
    卡尔曼平滑器用于趋势和残差的插补
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def smooth(self, data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        使用卡尔曼滤波和平滑处理缺失值

        Parameters:
        -----------
        data : np.ndarray
            输入数据，缺失值用 np.nan 表示
        missing_mask : np.ndarray
            缺失值掩码，True 表示缺失

        Returns:
        --------
        smoothed_data : np.ndarray
            平滑后的数据
        """
        # 将数据转移到 GPU
        data_tensor = torch.from_numpy(data).float().to(self.device)
        mask_tensor = torch.from_numpy(~missing_mask).float().to(self.device)

        n = len(data)

        # 状态空间模型参数
        # 状态: [level, trend]
        F = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=self.device)  # 状态转移矩阵
        H = torch.tensor([[1.0, 0.0]], device=self.device)  # 观测矩阵
        Q = torch.eye(2, device=self.device) * 0.01  # 过程噪声协方差
        R = torch.tensor([[1.0]], device=self.device)  # 观测噪声协方差

        # 初始化
        x = torch.zeros(2, device=self.device)  # 状态估计
        P = torch.eye(2, device=self.device) * 10  # 状态协方差

        # 存储前向滤波结果
        x_filtered = []
        P_filtered = []

        # 前向卡尔曼滤波
        for t in range(n):
            # 预测步骤
            x_pred = F @ x
            P_pred = F @ P @ F.t() + Q

            if not torch.isnan(data_tensor[t]) and mask_tensor[t] > 0:
                # 更新步骤（观测到数据）
                y = data_tensor[t].unsqueeze(0)
                y_pred = H @ x_pred
                innovation = y - y_pred
                S = H @ P_pred @ H.t() + R
                K = P_pred @ H.t() @ torch.inverse(S)

                x = x_pred + K.squeeze() * innovation
                P = P_pred - K @ S @ K.t()
            else:
                # 无观测数据，使用预测值
                x = x_pred
                P = P_pred

            x_filtered.append(x.clone())
            P_filtered.append(P.clone())

        # 后向RTS平滑
        x_smoothed = [x_filtered[-1]]
        P_smoothed = [P_filtered[-1]]

        for t in range(n-2, -1, -1):
            x_filt = x_filtered[t]
            P_filt = P_filtered[t]

            # 预测下一时刻
            x_pred = F @ x_filt
            P_pred = F @ P_filt @ F.t() + Q

            # RTS平滑增益
            C = P_filt @ F.t() @ torch.inverse(P_pred)

            # 平滑估计
            x_smooth = x_filt + C @ (x_smoothed[0] - x_pred)
            P_smooth = P_filt + C @ (P_smoothed[0] - P_pred) @ C.t()

            x_smoothed.insert(0, x_smooth)
            P_smoothed.insert(0, P_smooth)

        # 提取平滑后的水平值（level）
        smoothed = torch.stack([x[0] for x in x_smoothed])

        return smoothed.cpu().numpy()


class STLKalmanImputer:
    """
    STL + 卡尔曼平滑器缺失值填补
    """
    def __init__(self,
                 daily_period: int = 144,
                 weekly_period: int = 1008,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Parameters:
        -----------
        daily_period : int
            日季节性周期（10分钟间隔，144 = 24小时）
        weekly_period : int
            周季节性周期（10分钟间隔，1008 = 7天）
        device : str
            计算设备 'cuda' 或 'cpu'
        """
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.device = device
        self.kalman_smoother = KalmanSmoother(device=device)

    def detect_missing(self, series: pd.Series) -> np.ndarray:
        """
        检测缺失值和异常值

        Returns:
        --------
        missing_mask : np.ndarray
            True 表示缺失或异常
        """
        # 检测 NaN 和 Inf
        missing_mask = series.isna() | np.isinf(series)

        # 检测极端值（超过均值±5倍标准差）
        valid_data = series[~missing_mask]
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            extreme_mask = (series - mean_val).abs() > 5 * std_val
            missing_mask = missing_mask | extreme_mask

        return missing_mask.values

    def stl_decompose(self, series: pd.Series, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STL分解

        Returns:
        --------
        trend, seasonal, residual : tuple of np.ndarray
        """
        # 对于有缺失值的序列，先进行简单插值
        series_filled = series.interpolate(method='linear', limit_direction='both')

        # 如果仍有缺失值，用均值填充
        if series_filled.isna().any():
            series_filled = series_filled.fillna(series_filled.mean())

        try:
            # STL分解
            stl = STL(series_filled, period=period, seasonal=13)
            result = stl.fit()

            return result.trend.values, result.seasonal.values, result.resid.values
        except Exception as e:
            print(f"STL分解失败: {e}")
            # 返回简单的分解
            return (savgol_filter(series_filled, window_length=min(51, len(series_filled)//2*2+1), polyorder=2),
                    np.zeros_like(series_filled),
                    series_filled - savgol_filter(series_filled, window_length=min(51, len(series_filled)//2*2+1), polyorder=2))

    def impute_series(self, series: pd.Series, missing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        对单个时间序列进行缺失值填补

        Parameters:
        -----------
        series : pd.Series
            输入时间序列
        missing_mask : np.ndarray, optional
            缺失值掩码，如果为 None 则自动检测

        Returns:
        --------
        imputed : np.ndarray
            填补后的序列
        """
        if missing_mask is None:
            missing_mask = self.detect_missing(series)

        if not missing_mask.any():
            return series.values

        # 1. 提取日季节性
        print("  提取日季节性...")
        _, seasonal_daily, _ = self.stl_decompose(series, self.daily_period)

        # 2. 去除日季节性
        deseasonalized_daily = series.values - seasonal_daily

        # 3. 提取周季节性（如果数据足够长）
        if len(series) >= self.weekly_period * 2:
            print("  提取周季节性...")
            _, seasonal_weekly, _ = self.stl_decompose(
                pd.Series(deseasonalized_daily),
                self.weekly_period
            )
        else:
            seasonal_weekly = np.zeros_like(deseasonalized_daily)

        # 4. 去除周季节性得到趋势+残差
        trend_residual = deseasonalized_daily - seasonal_weekly

        # 5. 对趋势+残差使用卡尔曼平滑
        print("  卡尔曼平滑处理...")
        trend_residual_smoothed = self.kalman_smoother.smooth(
            trend_residual,
            missing_mask
        )

        # 6. 重组时间序列
        imputed = trend_residual_smoothed + seasonal_daily + seasonal_weekly

        return imputed

    def impute_dataframe(self, df: pd.DataFrame,
                        time_col: Optional[str] = None,
                        value_cols: Optional[list] = None) -> pd.DataFrame:
        """
        对DataFrame中的多个时间序列进行缺失值填补

        Parameters:
        -----------
        df : pd.DataFrame
            输入数据框，第一列为时间戳，其余列为各区域的交通流量
        time_col : str, optional
            时间列名称，如果为 None 则使用第一列
        value_cols : list, optional
            需要处理的列名列表，如果为 None 则处理除时间列外的所有列

        Returns:
        --------
        df_imputed : pd.DataFrame
            填补后的数据框
        """
        df_imputed = df.copy()

        # 确定时间列和值列
        if time_col is None:
            time_col = df.columns[0]
        if value_cols is None:
            value_cols = [col for col in df.columns if col != time_col]

        print(f"\n开始处理 {len(value_cols)} 个区域的数据...")
        print(f"使用设备: {self.device}")

        # 对每个区域进行处理
        for i, col in enumerate(value_cols):
            print(f"\n处理区域 [{i+1}/{len(value_cols)}]: {col}")

            series = df[col]
            missing_mask = self.detect_missing(series)

            if missing_mask.any():
                missing_rate = missing_mask.sum() / len(missing_mask) * 100
                print(f"  缺失率: {missing_rate:.2f}%")

                imputed = self.impute_series(series, missing_mask)
                df_imputed[col] = imputed
            else:
                print("  无缺失值")

        return df_imputed


def main():
    """
    主函数 - 演示使用
    """
    print("=" * 60)
    print("STL + 卡尔曼平滑器缺失值填补")
    print("=" * 60)

    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print("\n加载数据...")
    df_milano = pd.read_csv('../dataset/milano_traffic_nid.csv')
    df_trentino = pd.read_csv('../dataset/trentino_traffic_nid.csv')

    print(f"米兰数据: {df_milano.shape}")
    print(f"特伦蒂诺数据: {df_trentino.shape}")

    # 创建填补器
    imputer = STLKalmanImputer(
        daily_period=144,
        weekly_period=1008,
        device=device
    )

    # 处理米兰数据（示例：只处理前5个区域以节省时间）
    print("\n" + "=" * 60)
    print("处理米兰数据（前5个区域）")
    print("=" * 60)

    sample_cols = [df_milano.columns[0]] + list(df_milano.columns[1:6])
    df_milano_sample = df_milano[sample_cols]
    df_milano_imputed = imputer.impute_dataframe(df_milano_sample)

    # 保存结果
    output_path = 'milano_imputed_sample.csv'
    df_milano_imputed.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
