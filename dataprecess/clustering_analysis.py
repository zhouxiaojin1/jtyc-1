"""
聚类分析模块 - k-Shape（基于形状的时序聚类）

原理：
对每个区域的"标准化日轮廓（z-normalize的24小时×10分钟=144点）"进行形状相似度聚类，
按模式而非幅度分群。

适配性：适合找"早晚高峰形态差异"与"周末模式"，对尺度不敏感
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class KShape:
    """
    k-Shape时序聚类算法（GPU加速版本）
    """
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        max_iter : int
            最大迭代次数
        tol : float
            收敛阈值
        device : str
            计算设备
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        # Check for CUDA compatibility issues
        if device == 'cuda' and torch.cuda.is_available():
            try:
                # Test if CUDA kernels are compatible
                test_tensor = torch.zeros(1, device='cuda')
                _ = test_tensor + 1
                self.device = device
            except RuntimeError as e:
                if 'no kernel image is available' in str(e):
                    print("Warning: CUDA is available but incompatible with your GPU. Falling back to CPU.")
                    self.device = 'cpu'
                else:
                    raise
        else:
            self.device = device

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def z_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Z-标准化（沿时间维度）
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 避免除以零
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (x - mean) / std

    def sbd_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        基于形状的距离（Shape-Based Distance）
        使用归一化互相关系数

        Parameters:
        -----------
        x : torch.Tensor, shape (batch, length)
        y : torch.Tensor, shape (batch, length) or (length,)

        Returns:
        --------
        distance : torch.Tensor
        """
        # 确保y是2D张量
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # Z-标准化
        x_norm = self.z_normalize(x)
        y_norm = self.z_normalize(y)

        # 计算互相关（使用FFT加速）
        n = x_norm.shape[-1]

        # 零填充到2n-1
        x_padded = F.pad(x_norm, (0, n-1))
        y_padded = F.pad(y_norm, (0, n-1))

        # FFT卷积
        X_fft = torch.fft.rfft(x_padded, dim=-1)
        Y_fft = torch.fft.rfft(y_padded.flip(dims=[-1]), dim=-1)

        # 互相关
        cc = torch.fft.irfft(X_fft * Y_fft, n=2*n-1, dim=-1)

        # 找到最大互相关系数
        max_cc = cc.max(dim=-1)[0]

        # 归一化系数
        x_energy = (x_norm ** 2).sum(dim=-1)  # shape: (batch_x,)
        y_energy = (y_norm ** 2).sum(dim=-1)  # shape: (batch_y,)

        # 使用外积计算所有配对的分母
        if y_energy.dim() == 0:
            y_energy = y_energy.unsqueeze(0)
        if x_energy.dim() == 0:
            x_energy = x_energy.unsqueeze(0)

        # 计算分母: 对于每个 x[i] 和 y[j] 的配对
        if len(y_energy) == 1:
            denom = torch.sqrt(x_energy * y_energy.item())
        else:
            denom = torch.sqrt(x_energy.unsqueeze(1) * y_energy.unsqueeze(0))
            denom = denom.squeeze()

        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        # 距离 = 1 - 归一化互相关系数
        distance = 1 - (max_cc / denom)

        return distance

    def update_cluster_center(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        更新聚类中心（使用序列平均）

        Parameters:
        -----------
        sequences : torch.Tensor, shape (n_samples, length)

        Returns:
        --------
        center : torch.Tensor, shape (length,)
        """
        # 简单平均（也可以使用DBA - DTW Barycenter Averaging）
        center = sequences.mean(dim=0)
        return self.z_normalize(center.unsqueeze(0)).squeeze()

    def fit(self, X: np.ndarray) -> 'KShape':
        """
        拟合模型

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_timesteps)
            输入时间序列

        Returns:
        --------
        self
        """
        # 转换为torch tensor
        X_tensor = torch.from_numpy(X).float().to(self.device)
        n_samples = X_tensor.shape[0]

        # 随机初始化聚类中心
        indices = torch.randperm(n_samples)[:self.n_clusters]
        centroids = X_tensor[indices].clone()
        centroids = self.z_normalize(centroids)

        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        prev_inertia = float('inf')

        for iteration in range(self.max_iter):
            # E步：分配样本到最近的聚类中心
            distances = torch.zeros(n_samples, self.n_clusters, device=self.device)

            for k in range(self.n_clusters):
                distances[:, k] = self.sbd_distance(X_tensor, centroids[k])

            new_labels = distances.argmin(dim=1)

            # M步：更新聚类中心
            new_centroids = []
            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.sum() > 0:
                    cluster_sequences = X_tensor[mask]
                    new_center = self.update_cluster_center(cluster_sequences)
                    new_centroids.append(new_center)
                else:
                    # 如果某个簇为空，保持原中心
                    new_centroids.append(centroids[k])

            centroids = torch.stack(new_centroids)

            # 计算inertia（类内距离和）
            inertia = 0.0
            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.sum() > 0:
                    cluster_distances = self.sbd_distance(X_tensor[mask], centroids[k])
                    inertia += cluster_distances.sum().item()

            # 检查收敛
            if abs(prev_inertia - inertia) < self.tol:
                print(f"  收敛于第 {iteration+1} 次迭代")
                break

            labels = new_labels
            prev_inertia = inertia

        # 保存结果
        self.cluster_centers_ = centroids.cpu().numpy()
        self.labels_ = labels.cpu().numpy()
        self.inertia_ = inertia

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本的聚类标签

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_timesteps)

        Returns:
        --------
        labels : np.ndarray
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)
        centroids_tensor = torch.from_numpy(self.cluster_centers_).float().to(self.device)

        n_samples = X_tensor.shape[0]
        distances = torch.zeros(n_samples, self.n_clusters, device=self.device)

        for k in range(self.n_clusters):
            distances[:, k] = self.sbd_distance(X_tensor, centroids_tensor[k])

        labels = distances.argmin(dim=1).cpu().numpy()
        return labels


class TrafficPatternClustering:
    """
    交通模式聚类分析
    """
    def __init__(self,
                 daily_points: int = 144,
                 n_weeks: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Parameters:
        -----------
        daily_points : int
            每日数据点数（144 = 24小时 × 6个10分钟）
        n_weeks : int
            提取最近n周的数据
        device : str
            计算设备
        """
        self.daily_points = daily_points
        self.n_weeks = n_weeks
        self.device = device

    def extract_daily_profiles(self, series: pd.Series,
                               weekday: bool = True) -> np.ndarray:
        """
        提取日轮廓

        Parameters:
        -----------
        series : pd.Series
            时间序列
        weekday : bool
            True: 工作日, False: 周末

        Returns:
        --------
        profiles : np.ndarray, shape (n_days, daily_points)
        """
        # 重塑为日矩阵
        n_total_days = len(series) // self.daily_points
        if n_total_days == 0:
            return np.array([])

        # 裁剪到完整天数
        series_trimmed = series[:n_total_days * self.daily_points]
        daily_matrix = series_trimmed.values.reshape(n_total_days, self.daily_points)

        # 筛选工作日/周末（假设星期一=0，星期日=6）
        # 这里简化处理，实际应根据时间戳判断
        if weekday:
            # 假设前5天是工作日（实际应该根据日期判断）
            # 这里简化为取奇数索引行
            mask = np.arange(n_total_days) % 7 < 5
        else:
            mask = np.arange(n_total_days) % 7 >= 5

        profiles = daily_matrix[mask]

        # 只保留最近n_weeks周的数据
        n_days_to_keep = self.n_weeks * 7 if weekday else self.n_weeks * 2
        if len(profiles) > n_days_to_keep:
            profiles = profiles[-n_days_to_keep:]

        return profiles

    def find_optimal_k(self, profiles: np.ndarray,
                      k_range: range = range(2, 8)) -> Tuple[int, List[float]]:
        """
        使用轮廓系数法寻找最优k值

        Returns:
        --------
        optimal_k : int
        silhouette_scores : list
        """
        if len(profiles) < 2:
            return 2, []

        silhouette_scores = []

        for k in k_range:
            if k > len(profiles):
                break

            print(f"  评估 k={k}...")
            kshape = KShape(n_clusters=k, max_iter=50, device=self.device)
            kshape.fit(profiles)

            # 计算轮廓系数
            if len(np.unique(kshape.labels_)) > 1:
                # 计算距离矩阵（使用SBD距离）
                n = len(profiles)
                dist_matrix = np.zeros((n, n))

                profiles_tensor = torch.from_numpy(profiles).float().to(self.device)
                for i in range(n):
                    dists = kshape.sbd_distance(
                        profiles_tensor,
                        profiles_tensor[i]
                    )
                    dist_matrix[i] = dists.cpu().numpy()

                score = silhouette_score(dist_matrix, kshape.labels_, metric='precomputed')
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)

        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3

        return optimal_k, silhouette_scores

    def cluster_regions(self, df: pd.DataFrame,
                       time_col: Optional[str] = None,
                       value_cols: Optional[list] = None,
                       weekday: bool = True,
                       auto_k: bool = True,
                       n_clusters: int = 3) -> Tuple[dict, np.ndarray]:
        """
        对多个区域进行聚类

        Parameters:
        -----------
        df : pd.DataFrame
            输入数据框
        time_col : str, optional
            时间列名称
        value_cols : list, optional
            区域列名列表
        weekday : bool
            是否使用工作日数据
        auto_k : bool
            是否自动选择k值
        n_clusters : int
            如果不自动选择，使用的聚类数

        Returns:
        --------
        results : dict
            聚类结果
        all_profiles : np.ndarray
            所有日轮廓
        """
        # 确定列
        if time_col is None:
            time_col = df.columns[0]
        if value_cols is None:
            value_cols = [col for col in df.columns if col != time_col]

        print(f"\n提取 {len(value_cols)} 个区域的日轮廓...")
        print(f"类型: {'工作日' if weekday else '周末'}")

        # 提取所有区域的日轮廓
        all_profiles = []
        region_names = []

        for col in value_cols:
            series = df[col]
            profiles = self.extract_daily_profiles(series, weekday=weekday)

            if len(profiles) > 0:
                # 对每个区域，取平均日轮廓
                avg_profile = profiles.mean(axis=0)
                all_profiles.append(avg_profile)
                region_names.append(col)

        all_profiles = np.array(all_profiles)
        print(f"提取了 {len(all_profiles)} 个区域的日轮廓")

        # 自动选择k值
        if auto_k:
            print("\n寻找最优聚类数...")
            optimal_k, scores = self.find_optimal_k(all_profiles)
            print(f"最优k值: {optimal_k}")
            n_clusters = optimal_k

        # 聚类
        print(f"\n执行 k-Shape 聚类 (k={n_clusters})...")
        kshape = KShape(n_clusters=n_clusters, max_iter=100, device=self.device)
        kshape.fit(all_profiles)

        # 整理结果
        results = {
            'n_clusters': n_clusters,
            'labels': kshape.labels_,
            'cluster_centers': kshape.cluster_centers_,
            'region_names': region_names,
            'inertia': kshape.inertia_
        }

        # 统计每个簇的成员
        for k in range(n_clusters):
            mask = kshape.labels_ == k
            cluster_regions = [region_names[i] for i in range(len(region_names)) if mask[i]]
            results[f'cluster_{k}_regions'] = cluster_regions
            print(f"  簇 {k}: {len(cluster_regions)} 个区域")

        return results, all_profiles

    def visualize_clusters(self, results: dict,
                          all_profiles: np.ndarray,
                          save_path: Optional[str] = None):
        """
        可视化聚类结果
        """
        n_clusters = results['n_clusters']
        labels = results['labels']
        centers = results['cluster_centers']

        # 创建子图
        fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4*n_clusters))
        if n_clusters == 1:
            axes = [axes]

        # 时间标签（小时）
        hours = np.arange(self.daily_points) * 10 / 60

        for k in range(n_clusters):
            ax = axes[k]

            # 绘制该簇的所有轮廓
            mask = labels == k
            cluster_profiles = all_profiles[mask]

            for profile in cluster_profiles:
                ax.plot(hours, profile, alpha=0.3, color='gray', linewidth=0.5)

            # 绘制聚类中心
            ax.plot(hours, centers[k], color='red', linewidth=2, label='聚类中心')

            ax.set_title(f'簇 {k} - {mask.sum()} 个区域')
            ax.set_xlabel('时间 (小时)')
            ax.set_ylabel('标准化流量')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.close()


def main():
    """
    主函数 - 演示使用
    """
    print("=" * 60)
    print("k-Shape 时序聚类分析")
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

    # 创建聚类器
    clustering = TrafficPatternClustering(
        daily_points=144,
        n_weeks=4,
        device=device
    )

    # 工作日模式聚类
    print("\n" + "=" * 60)
    print("工作日模式聚类")
    print("=" * 60)

    results_weekday, profiles_weekday = clustering.cluster_regions(
        df_milano,
        weekday=True,
        auto_k=True
    )

    # 可视化
    clustering.visualize_clusters(
        results_weekday,
        profiles_weekday,
        save_path='weekday_clusters.png'
    )

    # 周末模式聚类
    print("\n" + "=" * 60)
    print("周末模式聚类")
    print("=" * 60)

    results_weekend, profiles_weekend = clustering.cluster_regions(
        df_milano,
        weekday=False,
        auto_k=True
    )

    # 可视化
    clustering.visualize_clusters(
        results_weekend,
        profiles_weekend,
        save_path='weekend_clusters.png'
    )

    # 保存聚类结果
    report_path = 'clustering_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("聚类分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("工作日模式聚类:\n")
        f.write(f"  聚类数: {results_weekday['n_clusters']}\n")
        f.write(f"  总惯性: {results_weekday['inertia']:.4f}\n\n")

        for k in range(results_weekday['n_clusters']):
            regions = results_weekday[f'cluster_{k}_regions']
            f.write(f"  簇 {k} ({len(regions)} 个区域):\n")
            for region in regions[:10]:  # 只列出前10个
                f.write(f"    - {region}\n")
            if len(regions) > 10:
                f.write(f"    - ... 还有 {len(regions)-10} 个区域\n")
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("周末模式聚类:\n")
        f.write(f"  聚类数: {results_weekend['n_clusters']}\n")
        f.write(f"  总惯性: {results_weekend['inertia']:.4f}\n\n")

        for k in range(results_weekend['n_clusters']):
            regions = results_weekend[f'cluster_{k}_regions']
            f.write(f"  簇 {k} ({len(regions)} 个区域):\n")
            for region in regions[:10]:
                f.write(f"    - {region}\n")
            if len(regions) > 10:
                f.write(f"    - ... 还有 {len(regions)-10} 个区域\n")
            f.write("\n")

    print(f"\n聚类报告已保存到: {report_path}")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
