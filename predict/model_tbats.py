# -*- coding: utf-8 -*-
"""
TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, and Seasonal components)

适用：多季节性时间序列、复杂周期模式（小时+日+周）、自动化预测
关键设置：
- use_box_cox: True（自动Box-Cox变换）
- use_trend: True（捕捉趋势）
- use_arma_errors: True（处理残差自相关）
- seasonal_periods: [144, 1008]（10分钟数据：144=1天, 1008=1周）

优点：自动处理多季节性、鲁棒性强、无需手动特征工程、适合复杂周期模式
缺点：训练速度较慢（尤其是多区域）、对长序列内存占用大、单变量模型（不利用区域关联）
"""

import numpy as np
import pandas as pd
from tbats import TBATS
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import pickle
from tqdm import tqdm
import os
import sys
from pathlib import Path
import time
from joblib import Parallel, delayed

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 配置中文字体
from utils.plot_config import setup_chinese_font, apply_plot_style
setup_chinese_font()


class TBATSForecaster:
    """
    TBATS时间序列预测器

    TBATS模型特别适合处理：
    1. 多季节性模式（如小时、日、周的季节性）
    2. 非线性趋势（通过Box-Cox变换）
    3. 复杂的时间模式
    """
    def __init__(self,
                 forecast_horizon: int = 1008,
                 seasonal_periods: Optional[List[int]] = None,
                 use_box_cox: bool = True,
                 use_trend: bool = True,
                 use_damped_trend: bool = False,
                 use_arma_errors: bool = True,
                 show_warnings: bool = False,
                 n_jobs: int = 1,
                 multiprocessing_start_method: str = 'spawn'):
        """
        Parameters:
        -----------
        forecast_horizon : int
            预测步数
        seasonal_periods : list, optional
            季节性周期列表。例如 [144, 1008] 表示日周期和周周期
            如果为None，模型将自动检测
        use_box_cox : bool
            是否使用Box-Cox变换
        use_trend : bool
            是否包含趋势成分
        use_damped_trend : bool
            是否使用阻尼趋势
        use_arma_errors : bool
            是否对误差使用ARMA模型
        show_warnings : bool
            是否显示警告
        n_jobs : int
            并行任务数（用于训练多个区域模型）
        multiprocessing_start_method : str
            多进程启动方法（'spawn' 或 'fork'）
        """
        self.forecast_horizon = forecast_horizon
        self.seasonal_periods = seasonal_periods if seasonal_periods else [144, 1008]  # 默认：日周期+周周期
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs
        self.multiprocessing_start_method = multiprocessing_start_method

        self.models = {}  # 存储每个区域的模型
        self.training_history = {}  # 存储训练历史

        print(f"[INFO] TBATS预测器初始化")
        print(f"  预测步数: {self.forecast_horizon}")
        print(f"  季节性周期: {self.seasonal_periods}")
        print(f"  Box-Cox变换: {self.use_box_cox}")
        print(f"  趋势成分: {self.use_trend}")
        print(f"  ARMA误差: {self.use_arma_errors}")

    def _create_tbats_model(self) -> TBATS:
        """
        创建TBATS模型实例

        Returns:
        --------
        estimator : TBATS
            TBATS模型实例
        """
        estimator = TBATS(
            seasonal_periods=self.seasonal_periods,
            use_box_cox=self.use_box_cox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=self.use_arma_errors,
            show_warnings=self.show_warnings,
            n_jobs=1,  # 单个模型内部使用单线程
            multiprocessing_start_method=self.multiprocessing_start_method
        )
        return estimator

    def _fit_single_region(self, region: str, data: np.ndarray) -> Tuple[str, object, Dict]:
        """
        训练单个区域的模型

        Parameters:
        -----------
        region : str
            区域名称
        data : np.ndarray
            时间序列数据

        Returns:
        --------
        tuple : (region, fitted_model, training_info)
        """
        try:
            start_time = time.time()

            # 检查数据有效性
            if len(data) < max(self.seasonal_periods) * 2:
                print(f"[WARNING] 区域 {region} 数据不足，跳过")
                return region, None, {'error': 'insufficient_data'}

            # 检查是否有过多的零值或NaN
            valid_ratio = np.sum(~np.isnan(data) & (data != 0)) / len(data)
            if valid_ratio < 0.3:
                print(f"[WARNING] 区域 {region} 有效数据不足30%，跳过")
                return region, None, {'error': 'too_many_zeros'}

            # 创建并训练模型
            estimator = self._create_tbats_model()
            fitted_model = estimator.fit(data)

            # 计算拟合误差
            fitted_values = fitted_model.y_hat
            train_mae = mean_absolute_error(data, fitted_values)
            train_rmse = np.sqrt(mean_squared_error(data, fitted_values))

            elapsed_time = time.time() - start_time

            training_info = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'training_time': elapsed_time,
                'params': str(fitted_model.params) if hasattr(fitted_model, 'params') else ''
            }

            return region, fitted_model, training_info

        except Exception as e:
            print(f"[ERROR] 区域 {region} 训练失败: {str(e)}")
            return region, None, {'error': str(e)}

    def fit(self, train_df: pd.DataFrame,
            time_col: str,
            value_cols: List[str],
            parallel: bool = True) -> Dict:
        """
        训练模型（每个区域一个独立的TBATS模型）

        Parameters:
        -----------
        train_df : pd.DataFrame
            训练数据
        time_col : str
            时间列
        value_cols : list
            区域列
        parallel : bool
            是否使用并行训练

        Returns:
        --------
        train_results : dict
            训练结果汇总
        """
        print("=" * 80)
        print("开始训练 TBATS 模型")
        print(f"区域数量: {len(value_cols)}")
        print(f"训练数据长度: {len(train_df)}")
        print(f"并行训练: {parallel} (n_jobs={self.n_jobs})")
        print("=" * 80)

        # 检查时间列
        if time_col not in train_df.columns:
            print(f"[WARNING] 时间列 '{time_col}' 不存在，使用第一列")
            time_col = train_df.columns[0]

        # 准备数据
        region_data = {col: train_df[col].values for col in value_cols}

        # 并行或串行训练
        if parallel and self.n_jobs > 1 and len(value_cols) > 1:
            print(f"\n[INFO] 使用并行训练（{self.n_jobs} 个进程）...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_region)(region, data)
                for region, data in tqdm(region_data.items(), desc="训练区域模型")
            )
        else:
            print(f"\n[INFO] 使用串行训练...")
            results = []
            for region, data in tqdm(region_data.items(), desc="训练区域模型"):
                result = self._fit_single_region(region, data)
                results.append(result)

        # 汇总结果
        train_results = {
            'total_regions': len(value_cols),
            'successful_regions': 0,
            'failed_regions': 0,
            'avg_train_mae': 0,
            'avg_train_rmse': 0,
            'total_training_time': 0,
            'region_details': []
        }

        successful_maes = []
        successful_rmses = []

        for region, model, info in results:
            self.training_history[region] = info

            if model is not None:
                self.models[region] = model
                train_results['successful_regions'] += 1

                if 'train_mae' in info:
                    successful_maes.append(info['train_mae'])
                    successful_rmses.append(info['train_rmse'])

                if 'training_time' in info:
                    train_results['total_training_time'] += info['training_time']

                train_results['region_details'].append({
                    'region': region,
                    'status': 'success',
                    'train_mae': info.get('train_mae', None),
                    'train_rmse': info.get('train_rmse', None),
                    'training_time': info.get('training_time', None)
                })
            else:
                train_results['failed_regions'] += 1
                train_results['region_details'].append({
                    'region': region,
                    'status': 'failed',
                    'error': info.get('error', 'unknown')
                })

        if len(successful_maes) > 0:
            train_results['avg_train_mae'] = np.mean(successful_maes)
            train_results['avg_train_rmse'] = np.mean(successful_rmses)

        print(f"\n训练完成!")
        print(f"  成功: {train_results['successful_regions']} 个区域")
        print(f"  失败: {train_results['failed_regions']} 个区域")
        print(f"  平均训练MAE: {train_results['avg_train_mae']:.2f}")
        print(f"  平均训练RMSE: {train_results['avg_train_rmse']:.2f}")
        print(f"  总训练时间: {train_results['total_training_time']:.1f} 秒")

        return train_results

    def predict(self, value_cols: List[str],
                steps: Optional[int] = None) -> pd.DataFrame:
        """
        预测

        Parameters:
        -----------
        value_cols : list
            要预测的区域列
        steps : int, optional
            预测步数（如果为None则使用forecast_horizon）

        Returns:
        --------
        predictions_df : pd.DataFrame
            预测结果，列为区域，行为时间步
        """
        if steps is None:
            steps = self.forecast_horizon

        print(f"\n开始预测 {steps} 步...")

        predictions = {}

        for region in tqdm(value_cols, desc="预测"):
            if region not in self.models:
                print(f"[WARNING] 区域 {region} 没有训练好的模型，跳过")
                predictions[region] = np.full(steps, np.nan)
                continue

            try:
                model = self.models[region]
                # TBATS预测
                forecast, _ = model.forecast(steps=steps)
                predictions[region] = forecast
            except Exception as e:
                print(f"[ERROR] 区域 {region} 预测失败: {str(e)}")
                predictions[region] = np.full(steps, np.nan)

        predictions_df = pd.DataFrame(predictions)

        return predictions_df

    def evaluate(self, test_df: pd.DataFrame,
                predictions_df: pd.DataFrame,
                time_col: str) -> pd.DataFrame:
        """
        评估预测结果

        Parameters:
        -----------
        test_df : pd.DataFrame
            测试数据
        predictions_df : pd.DataFrame
            预测结果
        time_col : str
            时间列

        Returns:
        --------
        metrics_df : pd.DataFrame
            评估指标
        """
        value_cols = [col for col in test_df.columns if col != time_col]

        metrics = []
        for region in value_cols:
            if region not in predictions_df.columns:
                continue

            y_true = test_df[region].values[:len(predictions_df)]
            y_pred = predictions_df[region].values

            # 过滤NaN值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() == 0:
                print(f"[WARNING] 区域 {region} 没有有效的预测数据")
                continue

            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            # 计算指标
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

            # MAPE处理零值
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100

            metrics.append({
                'region': region,
                'test_mae': mae,
                'test_rmse': rmse,
                'test_mape': mape,
                'valid_points': mask.sum()
            })

        metrics_df = pd.DataFrame(metrics)

        if len(metrics_df) > 0:
            print("\n" + "=" * 80)
            print("测试集评估结果:")
            print("=" * 80)
            print(f"平均MAE: {metrics_df['test_mae'].mean():.2f}")
            print(f"平均RMSE: {metrics_df['test_rmse'].mean():.2f}")
            print(f"平均MAPE: {metrics_df['test_mape'].mean():.2f}%")
            print(f"最佳区域 (MAE): {metrics_df.loc[metrics_df['test_mae'].idxmin(), 'region']} "
                  f"(MAE={metrics_df['test_mae'].min():.2f})")
            print(f"最差区域 (MAE): {metrics_df.loc[metrics_df['test_mae'].idxmax(), 'region']} "
                  f"(MAE={metrics_df['test_mae'].max():.2f})")

        return metrics_df

    def visualize_predictions(self, train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             predictions_df: pd.DataFrame,
                             time_col: str,
                             region_names: Optional[List[str]] = None,
                             n_regions: int = 3,
                             save_path: Optional[str] = None):
        """
        可视化预测结果

        Parameters:
        -----------
        train_df : pd.DataFrame
            训练数据
        test_df : pd.DataFrame
            测试数据
        predictions_df : pd.DataFrame
            预测结果
        time_col : str
            时间列
        region_names : list, optional
            要可视化的区域名称
        n_regions : int
            如果region_names为None，随机选择的区域数
        save_path : str, optional
            保存路径
        """
        from utils.plot_config import ensure_chinese_font
        ensure_chinese_font()

        value_cols = [col for col in train_df.columns if col != time_col]

        if region_names is None:
            # 随机选择区域
            available_regions = [r for r in value_cols if r in predictions_df.columns]
            region_names = np.random.choice(
                available_regions,
                min(n_regions, len(available_regions)),
                replace=False
            )

        n_plots = len(region_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5*n_plots))
        if n_plots == 1:
            axes = [axes]

        for idx, region in enumerate(region_names):
            ax = axes[idx]

            # 训练数据（显示最后1008个点）
            train_series = train_df[region].iloc[-1008:]
            train_time = np.arange(len(train_series))

            # 测试数据
            test_series = test_df[region].iloc[:len(predictions_df)]
            test_time = np.arange(len(train_series), len(train_series) + len(test_series))

            # 预测数据
            if region in predictions_df.columns:
                pred_series = predictions_df[region].values
                pred_time = test_time

                ax.plot(train_time, train_series.values,
                       label='训练数据', color='#2E86AB', alpha=0.6, linewidth=1.5)
                ax.plot(test_time, test_series.values,
                       label='真实值', color='#06A77D', linewidth=2.5)
                ax.plot(pred_time, pred_series,
                       label='TBATS预测', color='#D62828', linestyle='--', linewidth=2.5)

                # 计算该区域的误差
                mask = ~np.isnan(pred_series)
                if mask.sum() > 0:
                    mae = mean_absolute_error(test_series.values[mask], pred_series[mask])
                    ax.text(0.02, 0.98, f'MAE: {mae:.2f}',
                           transform=ax.transAxes, fontsize=11,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_title(f'区域: {region}', fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('时间步 (10分钟间隔)', fontsize=11)
            ax.set_ylabel('交通流量', fontsize=11)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axvline(x=len(train_series), color='black', linestyle=':', linewidth=1.5, alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] 可视化结果已保存到: {save_path}")

        plt.close()

    def plot_training_summary(self, save_path: Optional[str] = None):
        """
        绘制训练汇总信息

        Parameters:
        -----------
        save_path : str, optional
            保存路径
        """
        from utils.plot_config import ensure_chinese_font
        ensure_chinese_font()

        # 收集训练数据
        regions = []
        train_maes = []
        train_rmses = []
        training_times = []

        for region, info in self.training_history.items():
            if 'train_mae' in info and 'train_rmse' in info:
                regions.append(region)
                train_maes.append(info['train_mae'])
                train_rmses.append(info['train_rmse'])
                training_times.append(info.get('training_time', 0))

        if len(regions) == 0:
            print("[WARNING] 没有训练历史数据可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. MAE分布
        ax = axes[0, 0]
        ax.bar(range(len(train_maes)), train_maes, color='#2E86AB', alpha=0.7)
        ax.set_xlabel('区域索引', fontsize=11)
        ax.set_ylabel('训练MAE', fontsize=11)
        ax.set_title('各区域训练MAE', fontsize=13, fontweight='bold')
        ax.axhline(y=np.mean(train_maes), color='red', linestyle='--',
                  label=f'平均MAE: {np.mean(train_maes):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. RMSE分布
        ax = axes[0, 1]
        ax.bar(range(len(train_rmses)), train_rmses, color='#06A77D', alpha=0.7)
        ax.set_xlabel('区域索引', fontsize=11)
        ax.set_ylabel('训练RMSE', fontsize=11)
        ax.set_title('各区域训练RMSE', fontsize=13, fontweight='bold')
        ax.axhline(y=np.mean(train_rmses), color='red', linestyle='--',
                  label=f'平均RMSE: {np.mean(train_rmses):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 训练时间
        ax = axes[1, 0]
        ax.bar(range(len(training_times)), training_times, color='#F77F00', alpha=0.7)
        ax.set_xlabel('区域索引', fontsize=11)
        ax.set_ylabel('训练时间 (秒)', fontsize=11)
        ax.set_title('各区域训练时间', fontsize=13, fontweight='bold')
        ax.axhline(y=np.mean(training_times), color='red', linestyle='--',
                  label=f'平均时间: {np.mean(training_times):.1f}秒')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. MAE vs 训练时间散点图
        ax = axes[1, 1]
        scatter = ax.scatter(training_times, train_maes,
                           c=train_rmses, cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('训练时间 (秒)', fontsize=11)
        ax.set_ylabel('训练MAE', fontsize=11)
        ax.set_title('训练时间 vs MAE (颜色=RMSE)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='RMSE')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] 训练汇总图已保存到: {save_path}")

        plt.close()

    def save_model(self, save_path: str):
        """
        保存模型

        Parameters:
        -----------
        save_path : str
            保存路径
        """
        print(f"\n[INFO] 保存模型到 {save_path}")

        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'training_history': self.training_history,
                'config': {
                    'forecast_horizon': self.forecast_horizon,
                    'seasonal_periods': self.seasonal_periods,
                    'use_box_cox': self.use_box_cox,
                    'use_trend': self.use_trend,
                    'use_damped_trend': self.use_damped_trend,
                    'use_arma_errors': self.use_arma_errors
                }
            }, f)

        print(f"[OK] 模型保存成功（{len(self.models)} 个区域模型）")

    def load_model(self, load_path: str):
        """
        加载模型

        Parameters:
        -----------
        load_path : str
            加载路径
        """
        print(f"\n[INFO] 从 {load_path} 加载模型")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.models = data['models']
        self.training_history = data['training_history']

        config = data['config']
        self.forecast_horizon = config['forecast_horizon']
        self.seasonal_periods = config['seasonal_periods']
        self.use_box_cox = config['use_box_cox']
        self.use_trend = config['use_trend']
        self.use_damped_trend = config['use_damped_trend']
        self.use_arma_errors = config['use_arma_errors']

        print(f"[OK] 模型加载成功（{len(self.models)} 个区域模型）")


def main():
    """
    主函数：演示TBATS模型训练流程
    """
    print("=" * 80)
    print("TBATS 时间序列预测模型")
    print("=" * 80)

    # 数据路径
    data_path = project_root / 'dataset' / 'milano_traffic_nid.csv'
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)

    # 1. 加载数据
    print(f"\n1. 加载数据: {data_path}")
    df = pd.read_csv(data_path)

    print(f"[OK] 数据加载成功，形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()[:5]}...")

    # 2. 划分数据集
    print("\n2. 划分数据集...")
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # 选择部分区域进行演示（TBATS训练较慢，建议先用少量区域测试）
    time_col = df.columns[0]
    all_regions = [col for col in df.columns if col != time_col]

    # 选择前3个区域进行演示
    sample_regions = all_regions[:3]
    print(f"选择区域数: {len(sample_regions)}")
    print(f"训练集长度: {len(train_df)}, 测试集长度: {len(test_df)}")

    train_sample = train_df[[time_col] + sample_regions]
    test_sample = test_df[[time_col] + sample_regions]

    # 3. 创建并训练模型
    print("\n3. 创建并训练TBATS模型...")

    forecaster = TBATSForecaster(
        forecast_horizon=288,          # 预测2天（288个10分钟）
        seasonal_periods=[144, 1008],  # 日周期 + 周周期
        use_box_cox=True,              # 使用Box-Cox变换
        use_trend=True,                # 包含趋势
        use_damped_trend=False,        # 不使用阻尼趋势
        use_arma_errors=True,          # 对误差使用ARMA
        show_warnings=False,           # 不显示警告
        n_jobs=2,                      # 并行训练（2个进程）
        multiprocessing_start_method='spawn'
    )

    train_results = forecaster.fit(
        train_sample,
        time_col,
        sample_regions,
        parallel=True  # 使用并行训练
    )

    # 4. 预测
    print("\n4. 进行预测...")
    predictions = forecaster.predict(sample_regions, steps=288)

    # 保存预测结果
    pred_path = output_dir / 'tbats_predictions.csv'
    predictions.to_csv(pred_path, index=False)
    print(f"[OK] 预测结果保存到: {pred_path}")

    # 5. 评估
    print("\n5. 评估...")
    test_metrics = forecaster.evaluate(test_sample, predictions, time_col)

    # 保存评估结果
    metrics_path = output_dir / 'tbats_test_metrics.csv'
    test_metrics.to_csv(metrics_path, index=False)
    print(f"[OK] 评估结果保存到: {metrics_path}")

    # 6. 可视化
    print("\n6. 生成可视化...")

    # 预测结果可视化
    pred_plot_path = output_dir / 'tbats_predictions_plot.png'
    forecaster.visualize_predictions(
        train_sample, test_sample, predictions, time_col,
        region_names=sample_regions,  # 显示所有3个区域
        save_path=str(pred_plot_path)
    )

    # 训练汇总可视化
    summary_plot_path = output_dir / 'tbats_training_summary.png'
    forecaster.plot_training_summary(save_path=str(summary_plot_path))

    # 7. 保存模型
    model_path = output_dir / 'tbats_model.pkl'
    forecaster.save_model(str(model_path))

    print("\n" + "=" * 80)
    print("[OK] TBATS模型训练和预测完成！")
    print("=" * 80)
    print(f"\n所有文件已保存到: {output_dir}")
    print(f"  - 预测结果: {pred_path.name}")
    print(f"  - 评估指标: {metrics_path.name}")
    print(f"  - 预测可视化: {pred_plot_path.name}")
    print(f"  - 训练汇总: {summary_plot_path.name}")
    print(f"  - 模型文件: {model_path.name}")


if __name__ == '__main__':
    main()
