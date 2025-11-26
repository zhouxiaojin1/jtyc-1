"""
异常数据分析模块 - S-H-ESD（Seasonal Hybrid ESD，季节混合广义ESD）

原理：
对STL分解的残差部分应用ESD检验（基于MAD的稳健Z分数），
在保留季节性的前提下识别异常尖峰/跌落。

适配性：对强季节交通数据效果稳定，能同时检测双向异常（高/低）
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy import stats
from typing import Tuple, Optional, List
import torch
import warnings
warnings.filterwarnings('ignore')


class SeasonalHybridESD:
    """
    季节混合广义ESD异常检测
    """
    def __init__(self,
                 period: int = 144,
                 max_anoms: float = 0.05,
                 alpha: float = 0.05,
                 direction: str = 'both',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Parameters:
        -----------
        period : int
            季节性周期（默认144 = 1天，10分钟间隔）
        max_anoms : float
            最大异常比例（0-1之间，默认5%）
        alpha : float
            显著性水平（默认0.05）
        direction : str
            检测方向：'both'（双向）, 'pos'（正向）, 'neg'（负向）
        device : str
            计算设备
        """
        self.period = period
        self.max_anoms = max_anoms
        self.alpha = alpha
        self.direction = direction
        self.device = device

    def stl_decompose(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STL分解

        Returns:
        --------
        trend, seasonal, residual : tuple of np.ndarray
        """
        # 处理缺失值
        series_filled = series.interpolate(method='linear', limit_direction='both')
        if series_filled.isna().any():
            series_filled = series_filled.fillna(series_filled.mean())

        try:
            # STL分解
            stl = STL(series_filled, period=self.period, seasonal=13, robust=True)
            result = stl.fit()
            return result.trend.values, result.seasonal.values, result.resid.values
        except Exception as e:
            print(f"  STL分解失败: {e}")
            # 简单分解
            from scipy.signal import savgol_filter
            n = len(series_filled)
            window = min(51, n//2*2+1)
            trend = savgol_filter(series_filled, window_length=window, polyorder=2)
            residual = series_filled - trend
            return trend, np.zeros_like(series_filled), residual

    def median_absolute_deviation(self, data: np.ndarray) -> float:
        """
        计算中位数绝对偏差（MAD）
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return mad

    def robust_z_score(self, data: np.ndarray) -> np.ndarray:
        """
        基于MAD的稳健Z分数
        """
        median = np.median(data)
        mad = self.median_absolute_deviation(data)

        # 避免除以零
        if mad == 0:
            mad = np.std(data)
            if mad == 0:
                return np.zeros_like(data)

        # 修正因子（使MAD等同于标准差）
        modified_z_scores = 0.6745 * (data - median) / mad
        return modified_z_scores

    def generalized_esd_test(self, residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        广义ESD检验

        Returns:
        --------
        anomaly_indices : np.ndarray
            异常值的索引
        anomaly_scores : np.ndarray
            异常值的分数
        """
        n = len(residuals)
        max_outliers = int(np.ceil(n * self.max_anoms))

        if max_outliers == 0:
            return np.array([]), np.array([])

        # 使用GPU加速计算
        residuals_tensor = torch.from_numpy(residuals.copy()).float().to(self.device)

        anomaly_indices = []
        anomaly_scores = []

        # 工作副本
        working_data = residuals.copy()

        for i in range(max_outliers):
            if len(working_data) < 3:
                break

            # 计算稳健Z分数
            z_scores = self.robust_z_score(working_data)

            # 根据方向选择最大值
            if self.direction == 'pos':
                max_idx = np.argmax(z_scores)
                max_z = z_scores[max_idx]
            elif self.direction == 'neg':
                max_idx = np.argmin(z_scores)
                max_z = abs(z_scores[max_idx])
            else:  # 'both'
                max_idx = np.argmax(np.abs(z_scores))
                max_z = abs(z_scores[max_idx])

            # ESD临界值
            n_curr = len(working_data)
            p = 1.0 - self.alpha / (2 * (n_curr - i))
            t_dist = stats.t.ppf(p, n_curr - i - 2)
            lambda_critical = ((n_curr - i - 1) * t_dist) / \
                            np.sqrt((n_curr - i - 2 + t_dist**2) * (n_curr - i))

            # 判断是否为异常
            if max_z > lambda_critical:
                # 找到原始索引
                original_idx = np.where(residuals == working_data[max_idx])[0]
                if len(original_idx) > 0:
                    original_idx = original_idx[0]
                    anomaly_indices.append(original_idx)
                    anomaly_scores.append(max_z)

                    # 从工作数据中移除
                    working_data = np.delete(working_data, max_idx)
            else:
                break

        return np.array(anomaly_indices), np.array(anomaly_scores)

    def detect_anomalies(self, series: pd.Series) -> Tuple[np.ndarray, dict]:
        """
        检测时间序列中的异常值

        Parameters:
        -----------
        series : pd.Series
            输入时间序列

        Returns:
        --------
        anomaly_mask : np.ndarray
            异常值掩码（True表示异常）
        info : dict
            包含详细信息的字典
        """
        # STL分解
        trend, seasonal, residual = self.stl_decompose(series)

        # 对残差应用ESD检验
        anomaly_indices, anomaly_scores = self.generalized_esd_test(residual)

        # 创建异常掩码
        anomaly_mask = np.zeros(len(series), dtype=bool)
        if len(anomaly_indices) > 0:
            anomaly_mask[anomaly_indices] = True

        # 收集信息
        info = {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': anomaly_scores,
            'anomaly_values': series.values[anomaly_indices] if len(anomaly_indices) > 0 else np.array([])
        }

        return anomaly_mask, info

    def replace_anomalies(self, series: pd.Series,
                         anomaly_mask: np.ndarray,
                         info: dict,
                         method: str = 'seasonal_median') -> np.ndarray:
        """
        替换异常值

        Parameters:
        -----------
        series : pd.Series
            原始序列
        anomaly_mask : np.ndarray
            异常值掩码
        info : dict
            检测信息
        method : str
            替换方法：
            - 'seasonal_median': 使用相邻时段的季节均值
            - 'interpolate': 线性插值
            - 'stl_reconstruct': 使用STL重构（趋势+季节）

        Returns:
        --------
        cleaned : np.ndarray
            清理后的序列
        """
        cleaned = series.values.copy()

        if not anomaly_mask.any():
            return cleaned

        anomaly_indices = np.where(anomaly_mask)[0]

        if method == 'seasonal_median':
            # 使用季节均值替换
            seasonal = info['seasonal']
            trend = info['trend']

            for idx in anomaly_indices:
                # 找到相同季节位置的其他时间点
                season_pos = idx % self.period
                same_season_indices = np.arange(season_pos, len(series), self.period)
                # 排除异常点
                same_season_indices = same_season_indices[~anomaly_mask[same_season_indices]]

                if len(same_season_indices) > 0:
                    # 使用中位数
                    replacement = np.median(series.values[same_season_indices])
                else:
                    # 使用趋势+季节
                    replacement = trend[idx] + seasonal[idx]

                cleaned[idx] = replacement

        elif method == 'interpolate':
            # 线性插值
            cleaned_series = pd.Series(cleaned)
            cleaned_series[anomaly_mask] = np.nan
            cleaned_series = cleaned_series.interpolate(method='linear', limit_direction='both')
            cleaned = cleaned_series.values

        elif method == 'stl_reconstruct':
            # 使用STL重构
            trend = info['trend']
            seasonal = info['seasonal']
            cleaned[anomaly_mask] = trend[anomaly_mask] + seasonal[anomaly_mask]

        return cleaned

    def detect_and_clean_dataframe(self,
                                   df: pd.DataFrame,
                                   time_col: Optional[str] = None,
                                   value_cols: Optional[list] = None,
                                   replace_method: str = 'seasonal_median') -> Tuple[pd.DataFrame, dict]:
        """
        对DataFrame中的多个时间序列进行异常检测和清理

        Parameters:
        -----------
        df : pd.DataFrame
            输入数据框
        time_col : str, optional
            时间列名称
        value_cols : list, optional
            需要处理的列名列表
        replace_method : str
            异常值替换方法

        Returns:
        --------
        df_cleaned : pd.DataFrame
            清理后的数据框
        detection_results : dict
            每个区域的检测结果
        """
        df_cleaned = df.copy()
        detection_results = {}

        # 确定时间列和值列
        if time_col is None:
            time_col = df.columns[0]
        if value_cols is None:
            value_cols = [col for col in df.columns if col != time_col]

        print(f"\n开始检测 {len(value_cols)} 个区域的异常值...")
        print(f"使用设备: {self.device}")
        print(f"参数: period={self.period}, max_anoms={self.max_anoms*100}%, alpha={self.alpha}, direction={self.direction}")

        for i, col in enumerate(value_cols):
            print(f"\n处理区域 [{i+1}/{len(value_cols)}]: {col}")

            series = df[col]

            # 检测异常
            anomaly_mask, info = self.detect_anomalies(series)

            n_anomalies = anomaly_mask.sum()
            anomaly_rate = n_anomalies / len(series) * 100

            print(f"  检测到 {n_anomalies} 个异常值 ({anomaly_rate:.2f}%)")

            if n_anomalies > 0:
                # 替换异常值
                cleaned = self.replace_anomalies(series, anomaly_mask, info, method=replace_method)
                df_cleaned[col] = cleaned

                # 保存检测结果
                detection_results[col] = {
                    'n_anomalies': n_anomalies,
                    'anomaly_rate': anomaly_rate,
                    'anomaly_indices': info['anomaly_indices'],
                    'anomaly_scores': info['anomaly_scores'],
                    'anomaly_values': info['anomaly_values']
                }

        return df_cleaned, detection_results


def main():
    """
    主函数 - 演示使用
    """
    print("=" * 60)
    print("S-H-ESD 季节混合广义ESD异常检测")
    print("=" * 60)

    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print("\n加载数据...")
    df_milano = pd.read_csv('../dataset/milano_traffic_nid.csv')

    print(f"米兰数据: {df_milano.shape}")

    # 创建检测器
    detector = SeasonalHybridESD(
        period=144,  # 日周期
        max_anoms=0.05,  # 最大5%异常
        alpha=0.05,
        direction='both',
        device=device
    )

    # 处理数据（示例：只处理前5个区域）
    print("\n" + "=" * 60)
    print("处理米兰数据（前5个区域）")
    print("=" * 60)

    sample_cols = [df_milano.columns[0]] + list(df_milano.columns[1:6])
    df_milano_sample = df_milano[sample_cols]

    df_cleaned, results = detector.detect_and_clean_dataframe(
        df_milano_sample,
        replace_method='seasonal_median'
    )

    # 保存结果
    output_path = 'milano_anomaly_cleaned_sample.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n清理后的数据已保存到: {output_path}")

    # 保存检测报告
    report_path = 'anomaly_detection_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("异常检测报告\n")
        f.write("=" * 60 + "\n\n")

        for col, info in results.items():
            f.write(f"区域: {col}\n")
            f.write(f"  异常数量: {info['n_anomalies']}\n")
            f.write(f"  异常率: {info['anomaly_rate']:.2f}%\n")
            f.write(f"  异常索引: {info['anomaly_indices'][:10]}...\n")  # 只显示前10个
            f.write(f"  异常分数: {info['anomaly_scores'][:10]}...\n\n")

    print(f"\n检测报告已保存到: {report_path}")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
