"""
时间序列聚类分析

功能：
1. K-Means 聚类
2. K-Shape 聚类（时间序列专用）
3. DTW（动态时间规整）聚类
4. 层次聚类
5. 聚类结果可视化和评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean
from typing import Optional, List, Dict, Tuple
import warnings
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 配置中文字体
from utils.plot_config import setup_chinese_font, apply_plot_style
setup_chinese_font()


class TimeSeriesClustering:
    """
    时间序列聚类分析器
    """

    def __init__(self, n_clusters: int = 5, method: str = 'kmeans',
                 normalize: bool = True, random_state: int = 42):
        """
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        method : str
            聚类方法：'kmeans', 'hierarchical', 'kshape'
        normalize : bool
            是否标准化数据
        random_state : int
            随机种子
        """
        self.n_clusters = n_clusters
        self.method = method
        self.normalize = normalize
        self.random_state = random_state

        self.scaler = StandardScaler() if normalize else None
        self.model = None
        self.labels = None
        self.cluster_centers = None
        self.metrics = {}

        print(f"[INFO] 聚类分析器初始化")
        print(f"  - 聚类数: {n_clusters}")
        print(f"  - 方法: {method}")
        print(f"  - 标准化: {normalize}")

    def _dtw_distance(self, ts1: np.ndarray, ts2: np.ndarray) -> float:
        """
        计算两个时间序列的DTW距离

        Parameters:
        -----------
        ts1, ts2 : np.ndarray
            时间序列

        Returns:
        --------
        distance : float
            DTW距离
        """
        n, m = len(ts1), len(ts2)
        dtw_matrix = np.zeros((n + 1, m + 1))

        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(ts1[i - 1] - ts2[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                             dtw_matrix[i, j - 1],
                                             dtw_matrix[i - 1, j - 1])

        return dtw_matrix[n, m]

    def _kshape_distance(self, ts1: np.ndarray, ts2: np.ndarray) -> float:
        """
        计算K-Shape距离（基于形状的距离）

        Parameters:
        -----------
        ts1, ts2 : np.ndarray
            时间序列

        Returns:
        --------
        distance : float
            K-Shape距离
        """
        # Z-normalization
        ts1_norm = (ts1 - np.mean(ts1)) / (np.std(ts1) + 1e-8)
        ts2_norm = (ts2 - np.mean(ts2)) / (np.std(ts2) + 1e-8)

        # 计算互相关
        cross_corr = np.correlate(ts1_norm, ts2_norm, mode='full')
        max_corr = np.max(cross_corr)

        # 距离 = 1 - 归一化的最大互相关
        distance = 1 - (max_corr / len(ts1))

        return distance

    def fit(self, data: pd.DataFrame, time_col: Optional[str] = None) -> np.ndarray:
        """
        执行聚类分析

        Parameters:
        -----------
        data : pd.DataFrame
            时间序列数据（每列是一个序列）
        time_col : str, optional
            时间列名（如果有）

        Returns:
        --------
        labels : np.ndarray
            聚类标签
        """
        print("\n" + "=" * 60)
        print("开始聚类分析")
        print("=" * 60)

        # 准备数据
        if time_col and time_col in data.columns:
            X = data.drop(columns=[time_col]).values.T  # (n_regions, n_timesteps)
        else:
            X = data.values.T  # (n_regions, n_timesteps)

        print(f"\n数据形状: {X.shape} (样本数={X.shape[0]}, 时间步={X.shape[1]})")

        # 标准化
        if self.normalize:
            print("[INFO] 标准化数据...")
            X = self.scaler.fit_transform(X)

        # 执行聚类
        print(f"[INFO] 使用 {self.method} 方法进行聚类...")

        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            self.labels = self.model.fit_predict(X)
            self.cluster_centers = self.model.cluster_centers_

        elif self.method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            self.labels = self.model.fit_predict(X)

            # 计算聚类中心
            self.cluster_centers = np.array([
                X[self.labels == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

        elif self.method == 'kshape':
            # K-Shape 聚类（简化实现）
            # 初始化中心
            np.random.seed(self.random_state)
            init_indices = np.random.choice(len(X), self.n_clusters, replace=False)
            self.cluster_centers = X[init_indices].copy()

            # 迭代优化
            max_iter = 50
            for iteration in range(max_iter):
                # 分配样本到最近的聚类
                distances = np.zeros((len(X), self.n_clusters))
                for i in range(len(X)):
                    for j in range(self.n_clusters):
                        distances[i, j] = self._kshape_distance(X[i], self.cluster_centers[j])

                old_labels = self.labels.copy() if self.labels is not None else None
                self.labels = np.argmin(distances, axis=1)

                # 更新聚类中心
                for i in range(self.n_clusters):
                    cluster_members = X[self.labels == i]
                    if len(cluster_members) > 0:
                        # 使用均值作为新中心
                        self.cluster_centers[i] = cluster_members.mean(axis=0)

                # 检查收敛
                if old_labels is not None and np.array_equal(self.labels, old_labels):
                    print(f"[INFO] K-Shape 收敛于第 {iteration + 1} 次迭代")
                    break

        else:
            raise ValueError(f"未知的聚类方法: {self.method}")

        # 计算评估指标
        self._calculate_metrics(X)

        print("\n" + "=" * 60)
        print("聚类完成!")
        print("=" * 60)
        self._print_cluster_summary()

        return self.labels

    def _calculate_metrics(self, X: np.ndarray):
        """计算聚类评估指标"""
        if len(np.unique(self.labels)) > 1:
            self.metrics['silhouette'] = silhouette_score(X, self.labels)
            self.metrics['davies_bouldin'] = davies_bouldin_score(X, self.labels)
            self.metrics['calinski_harabasz'] = calinski_harabasz_score(X, self.labels)
        else:
            self.metrics['silhouette'] = 0
            self.metrics['davies_bouldin'] = 0
            self.metrics['calinski_harabasz'] = 0

    def _print_cluster_summary(self):
        """打印聚类摘要"""
        print("\n聚类结果摘要:")
        print("-" * 60)

        for i in range(self.n_clusters):
            count = np.sum(self.labels == i)
            percentage = count / len(self.labels) * 100
            print(f"聚类 {i}: {count} 个样本 ({percentage:.1f}%)")

        print("-" * 60)
        print("\n评估指标:")
        print(f"  轮廓系数 (Silhouette): {self.metrics.get('silhouette', 0):.4f}")
        print(f"  Davies-Bouldin 指数: {self.metrics.get('davies_bouldin', 0):.4f}")
        print(f"  Calinski-Harabasz 指数: {self.metrics.get('calinski_harabasz', 0):.4f}")

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        获取聚类摘要信息

        Returns:
        --------
        summary_df : pd.DataFrame
            聚类摘要
        """
        summary = []
        for i in range(self.n_clusters):
            count = np.sum(self.labels == i)
            percentage = count / len(self.labels) * 100
            summary.append({
                'cluster': i,
                'count': count,
                'percentage': percentage
            })

        summary_df = pd.DataFrame(summary)
        return summary_df

    def plot_cluster_centers(self, save_path: Optional[str] = None):
        """
        绘制聚类中心

        Parameters:
        -----------
        save_path : str, optional
            保存路径
        """
        if self.cluster_centers is None:
            print("[ERROR] 没有聚类中心，请先执行聚类")
            return

        n_cols = min(3, self.n_clusters)
        n_rows = (self.n_clusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if self.n_clusters == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(self.n_clusters):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            count = np.sum(self.labels == i)
            ax.plot(self.cluster_centers[i], linewidth=2, color=f'C{i}')
            ax.set_title(f'聚类 {i} 中心\n({count} 个样本)', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('标准化值', fontsize=10)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(self.n_clusters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] 聚类中心图保存到 {save_path}")

        return fig

    def plot_cluster_distribution(self, save_path: Optional[str] = None):
        """
        绘制聚类分布（饼图和柱状图）

        Parameters:
        -----------
        save_path : str, optional
            保存路径
        """
        if self.labels is None:
            print("[ERROR] 没有聚类结果，请先执行聚类")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 饼图
        cluster_counts = [np.sum(self.labels == i) for i in range(self.n_clusters)]
        colors = [f'C{i}' for i in range(self.n_clusters)]

        axes[0].pie(cluster_counts, labels=[f'聚类 {i}' for i in range(self.n_clusters)],
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('聚类分布（饼图）', fontsize=14, fontweight='bold')

        # 柱状图
        axes[1].bar(range(self.n_clusters), cluster_counts, color=colors)
        axes[1].set_xlabel('聚类', fontsize=12)
        axes[1].set_ylabel('样本数', fontsize=12)
        axes[1].set_title('聚类分布（柱状图）', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(self.n_clusters))
        axes[1].set_xticklabels([f'聚类 {i}' for i in range(self.n_clusters)])
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] 聚类分布图保存到 {save_path}")

        return fig

    def plot_pca_visualization(self, data: pd.DataFrame, time_col: Optional[str] = None,
                              save_path: Optional[str] = None):
        """
        使用PCA降维可视化聚类结果

        Parameters:
        -----------
        data : pd.DataFrame
            原始数据
        time_col : str, optional
            时间列名
        save_path : str, optional
            保存路径
        """
        if self.labels is None:
            print("[ERROR] 没有聚类结果，请先执行聚类")
            return

        # 准备数据
        if time_col and time_col in data.columns:
            X = data.drop(columns=[time_col]).values.T
        else:
            X = data.values.T

        # PCA降维到2D
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        # 绘图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        for i in range(self.n_clusters):
            mask = self.labels == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      label=f'聚类 {i} ({np.sum(mask)})',
                      alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=12)
        ax.set_title('PCA 降维可视化', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] PCA可视化图保存到 {save_path}")

        return fig

    def plot_dendrogram(self, data: pd.DataFrame, time_col: Optional[str] = None,
                       save_path: Optional[str] = None):
        """
        绘制层次聚类树状图

        Parameters:
        -----------
        data : pd.DataFrame
            原始数据
        time_col : str, optional
            时间列名
        save_path : str, optional
            保存路径
        """
        # 准备数据
        if time_col and time_col in data.columns:
            X = data.drop(columns=[time_col]).values.T
            region_names = [col for col in data.columns if col != time_col]
        else:
            X = data.values.T
            region_names = list(data.columns)

        # 计算层次聚类
        Z = linkage(X, method='ward')

        # 绘图
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        dendrogram(Z, labels=region_names, ax=ax, leaf_font_size=8)
        ax.set_xlabel('区域', fontsize=12)
        ax.set_ylabel('距离', fontsize=12)
        ax.set_title('层次聚类树状图', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] 树状图保存到 {save_path}")

        return fig

    def find_optimal_k(self, data: pd.DataFrame, time_col: Optional[str] = None,
                      k_range: range = range(2, 11), save_path: Optional[str] = None):
        """
        寻找最优聚类数（肘部法则和轮廓系数）

        Parameters:
        -----------
        data : pd.DataFrame
            数据
        time_col : str, optional
            时间列名
        k_range : range
            k值范围
        save_path : str, optional
            保存路径
        """
        print("\n" + "=" * 60)
        print("寻找最优聚类数")
        print("=" * 60)

        # 准备数据
        if time_col and time_col in data.columns:
            X = data.drop(columns=[time_col]).values.T
        else:
            X = data.values.T

        if self.normalize:
            X = self.scaler.fit_transform(X)

        inertias = []
        silhouettes = []

        for k in tqdm(k_range, desc="测试不同的k值"):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 肘部法则
        axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('聚类数 (k)', fontsize=12)
        axes[0].set_ylabel('惯性 (Inertia)', fontsize=12)
        axes[0].set_title('肘部法则', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 轮廓系数
        axes[1].plot(k_range, silhouettes, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('聚类数 (k)', fontsize=12)
        axes[1].set_ylabel('轮廓系数', fontsize=12)
        axes[1].set_title('轮廓系数法', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 标记最优k
        best_k = list(k_range)[np.argmax(silhouettes)]
        axes[1].axvline(x=best_k, color='red', linestyle='--', linewidth=2,
                       label=f'最优 k={best_k}')
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] 最优k图保存到 {save_path}")

        print(f"\n推荐的最优聚类数: k = {best_k} (轮廓系数 = {max(silhouettes):.4f})")

        return fig, best_k


def main():
    """主函数：演示聚类分析"""
    print("=" * 60)
    print("时间序列聚类分析")
    print("=" * 60)

    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'dataset', 'milano_traffic_nid.csv')
    df = pd.read_csv(data_path)

    print(f"\n数据形状: {df.shape}")

    # 选择部分区域和时间段进行演示
    time_col = df.columns[0]
    all_regions = [col for col in df.columns if col != time_col]
    sample_regions = all_regions[:20]  # 选择20个区域

    # 选择一周的数据
    sample_data = df[[time_col] + sample_regions].iloc[:1008]  # 一周数据

    print(f"选择区域数: {len(sample_regions)}")
    print(f"时间步数: {len(sample_data)}")

    # 创建输出目录
    output_dir = os.path.join(script_dir, '..', 'output', 'clustering')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 寻找最优聚类数
    clustering = TimeSeriesClustering(n_clusters=5, method='kmeans', normalize=True)
    fig, best_k = clustering.find_optimal_k(
        sample_data,
        time_col=time_col,
        k_range=range(2, 11),
        save_path=os.path.join(output_dir, 'optimal_k.png')
    )

    # 2. 使用最优k进行聚类
    clustering = TimeSeriesClustering(n_clusters=best_k, method='kmeans', normalize=True)
    labels = clustering.fit(sample_data, time_col=time_col)

    # 3. 保存聚类结果
    cluster_result = pd.DataFrame({
        'region': sample_regions,
        'cluster': labels
    })
    cluster_result.to_csv(os.path.join(output_dir, 'cluster_result.csv'), index=False)

    # 4. 可视化
    clustering.plot_cluster_centers(save_path=os.path.join(output_dir, 'cluster_centers.png'))
    clustering.plot_cluster_distribution(save_path=os.path.join(output_dir, 'cluster_distribution.png'))
    clustering.plot_pca_visualization(sample_data, time_col=time_col,
                                     save_path=os.path.join(output_dir, 'pca_visualization.png'))
    clustering.plot_dendrogram(sample_data, time_col=time_col,
                              save_path=os.path.join(output_dir, 'dendrogram.png'))

    # 5. 保存聚类摘要
    summary = clustering.get_cluster_summary()
    summary.to_csv(os.path.join(output_dir, 'cluster_summary.csv'), index=False)

    print("\n" + "=" * 60)
    print("[OK] 聚类分析完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
