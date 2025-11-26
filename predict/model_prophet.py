# -*- coding: utf-8 -*-
"""
Prophet (Facebook时间序列预测算法)

适用：复杂季节性模式、节假日效应、趋势变化、多周期时间序列
关键设置：
- seasonality_mode: 'multiplicative'（乘法季节性，适合交通流量）
- daily_seasonality: True（日周期）
- weekly_seasonality: True（周周期）
- yearly_seasonality: False（数据时长不足1年可关闭）

优点：
1. 速度快（比TBATS快10-100倍）
2. 自动处理多季节性和趋势
3. 支持节假日和特殊事件
4. 缺失值自动处理
5. 提供不确定性区间
6. 可解释性强（趋势+季节性+残差分解）

缺点：
1. 单变量模型（不利用区域间关联）
2. 对突发事件敏感度较低
3. 需要足够历史数据（建议>2周）
"""

import numpy as np
import pandas as pd
from prophet import Prophet
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
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 配置中文字体
from utils.plot_config import setup_chinese_font, apply_plot_style
setup_chinese_font()


class ProphetForecaster:
    """
    Prophet时间序列预测器

    Prophet模型特别适合处理：
    1. 强季节性模式（日、周、月、年）
    2. 趋势变化点自动检测
    3. 节假日和特殊事件
    4. 缺失值和异常值
    """
    def __init__(self,
                 forecast_horizon: int = 1008,
                 seasonality_mode: str = 'multiplicative',
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 interval_width: float = 0.95,
                 n_jobs: int = 1):
        """
        Parameters:
        -----------
        forecast_horizon : int
            预测步数
        seasonality_mode : str
            季节性模式：'additive'（加法）或 'multiplicative'（乘法）
            交通流量建议使用multiplicative
        daily_seasonality : bool
            是否包含日周期
        weekly_seasonality : bool
            是否包含周周期
        yearly_seasonality : bool
            是否包含年周期（需要至少1年数据）
        changepoint_prior_scale : float
            趋势变化点的灵活度（0.001-0.5，越大越灵活）
        seasonality_prior_scale : float
            季节性的灵活度（0.01-10，越大越灵活）
        interval_width : float
            预测区间宽度（置信度）
        n_jobs : int
            并行训练任务数
        """
        self.forecast_horizon = forecast_horizon
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.n_jobs = n_jobs

        self.models = {}  # 存储每个区域的模型
        self.training_history = {}  # 存储训练历史

        print(f"[INFO] Prophet预测器初始化")
        print(f"  预测步数: {self.forecast_horizon}")
        print(f"  季节性模式: {self.seasonality_mode}")
        print(f"  日周期: {self.daily_seasonality}")
        print(f"  周周期: {self.weekly_seasonality}")
        print(f"  趋势变化点灵活度: {self.changepoint_prior_scale}")
        print(f"  季节性灵活度: {self.seasonality_prior_scale}")

    def _create_prophet_model(self) -> Prophet:
        """
        创建Prophet模型实例

        Returns:
        --------
        model : Prophet
            Prophet模型实例
        """
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            uncertainty_samples=0  # 加速预测（不计算不确定性）
        )

        # 添加10分钟周期的季节性（144个点=1天）
        # 使用傅里叶阶数来捕捉复杂的日内模式
        model.add_seasonality(
            name='10min',
            period=1,  # 1天（Prophet内部会转换为小时）
            fourier_order=20  # 傅里叶阶数（越大越复杂）
        )

        return model

    def _prepare_prophet_data(self, time_series: pd.Series, start_time: pd.Timestamp) -> pd.DataFrame:
        """
        准备Prophet格式的数据（ds, y列）

        Parameters:
        -----------
        time_series : pd.Series
            时间序列数据
        start_time : pd.Timestamp
            起始时间

        Returns:
        --------
        prophet_df : pd.DataFrame
            Prophet格式数据（包含ds和y列）
        """
        # 生成10分钟间隔的时间戳
        timestamps = pd.date_range(
            start=start_time,
            periods=len(time_series),
            freq='10min'
        )

        prophet_df = pd.DataFrame({
            'ds': timestamps,
            'y': time_series.values
        })

        # 移除NaN值（Prophet不接受NaN）
        prophet_df = prophet_df.dropna()

        return prophet_df

    def _fit_single_region(self, region: str, data: np.ndarray,
                          start_time: pd.Timestamp) -> Tuple[str, object, Dict]:
        """
        训练单个区域的模型

        Parameters:
        -----------
        region : str
            区域名称
        data : np.ndarray
            时间序列数据
        start_time : pd.Timestamp
            起始时间

        Returns:
        --------
        tuple : (region, fitted_model, training_info)
        """
        try:
            start_train_time = time.time()

            # 检查数据有效性
            valid_ratio = np.sum(~np.isnan(data) & (data > 0)) / len(data)
            if valid_ratio < 0.3:
                print(f"[WARNING] 区域 {region} 有效数据不足30%，跳过")
                return region, None, {'error': 'insufficient_valid_data'}

            # 准备Prophet数据格式
            time_series = pd.Series(data)
            prophet_df = self._prepare_prophet_data(time_series, start_time)

            if len(prophet_df) < 288:  # 至少需要2天数据
                print(f"[WARNING] 区域 {region} 有效数据点不足288，跳过")
                return region, None, {'error': 'too_few_data_points'}

            # 创建并训练模型
            model = self._create_prophet_model()

            # 禁用Prophet的日志输出
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)

            model.fit(prophet_df)

            # 计算拟合误差（在训练集上）
            train_forecast = model.predict(prophet_df[['ds']])
            y_true = prophet_df['y'].values
            y_pred = train_forecast['yhat'].values

            train_mae = mean_absolute_error(y_true, y_pred)
            train_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            elapsed_time = time.time() - start_train_time

            training_info = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'training_time': elapsed_time,
                'data_points': len(prophet_df),
                'changepoints': len(model.changepoints)
            }

            return region, model, training_info

        except Exception as e:
            print(f"[ERROR] 区域 {region} 训练失败: {str(e)}")
            return region, None, {'error': str(e)}

    def fit(self, train_df: pd.DataFrame,
            time_col: str,
            value_cols: List[str],
            start_time: Optional[str] = None,
            parallel: bool = True) -> Dict:
        """
        训练模型（每个区域一个独立的Prophet模型）

        Parameters:
        -----------
        train_df : pd.DataFrame
            训练数据
        time_col : str
            时间列
        value_cols : list
            区域列
        start_time : str, optional
            起始时间（如'2013-11-01 00:00:00'），如果为None则使用train_df的时间列
        parallel : bool
            是否使用并行训练

        Returns:
        --------
        train_results : dict
            训练结果汇总
        """
        print("=" * 80)
        print("开始训练 Prophet 模型")
        print(f"区域数量: {len(value_cols)}")
        print(f"训练数据长度: {len(train_df)}")
        print(f"并行训练: {parallel} (n_jobs={self.n_jobs})")
        print("=" * 80)

        # 确定起始时间
        if start_time is None:
            if time_col in train_df.columns:
                start_time = pd.to_datetime(train_df[time_col].iloc[0])
            else:
                start_time = pd.Timestamp('2013-11-01 00:00:00')
                print(f"[WARNING] 未指定起始时间，使用默认值: {start_time}")
        else:
            start_time = pd.Timestamp(start_time)

        print(f"起始时间: {start_time}")

        # 准备数据
        region_data = {col: train_df[col].values for col in value_cols}

        # 并行或串行训练
        if parallel and self.n_jobs > 1 and len(value_cols) > 1:
            print(f"\n[INFO] 使用并行训练（{self.n_jobs} 个进程）...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_region)(region, data, start_time)
                for region, data in tqdm(region_data.items(), desc="训练区域模型")
            )
        else:
            print(f"\n[INFO] 使用串行训练...")
            results = []
            for region, data in tqdm(region_data.items(), desc="训练区域模型"):
                result = self._fit_single_region(region, data, start_time)
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
                    'training_time': info.get('training_time', None),
                    'data_points': info.get('data_points', None),
                    'changepoints': info.get('changepoints', None)
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
        print(f"  平均每区域训练时间: {train_results['total_training_time']/max(train_results['successful_regions'],1):.2f} 秒")

        return train_results

    def predict(self, value_cols: List[str],
                last_train_time: pd.Timestamp,
                steps: Optional[int] = None,
                return_components: bool = False) -> pd.DataFrame:
        """
        预测

        Parameters:
        -----------
        value_cols : list
            要预测的区域列
        last_train_time : pd.Timestamp
            训练集最后时间点
        steps : int, optional
            预测步数（如果为None则使用forecast_horizon）
        return_components : bool
            是否返回趋势、季节性等成分

        Returns:
        --------
        predictions_df : pd.DataFrame
            预测结果，列为区域，行为时间步
        """
        if steps is None:
            steps = self.forecast_horizon

        print(f"\n开始预测 {steps} 步...")

        # 创建未来时间点
        future_dates = pd.date_range(
            start=last_train_time + pd.Timedelta(minutes=10),
            periods=steps,
            freq='10min'
        )

        predictions = {}
        components_dict = {} if return_components else None

        for region in tqdm(value_cols, desc="预测"):
            if region not in self.models:
                print(f"[WARNING] 区域 {region} 没有训练好的模型，跳过")
                predictions[region] = np.full(steps, np.nan)
                continue

            try:
                model = self.models[region]

                # 创建未来数据框
                future = pd.DataFrame({'ds': future_dates})

                # Prophet预测
                forecast = model.predict(future)
                predictions[region] = forecast['yhat'].values

                # 如果需要返回成分
                if return_components:
                    components_dict[region] = {
                        'trend': forecast['trend'].values,
                        'weekly': forecast.get('weekly', np.zeros(steps)).values if self.weekly_seasonality else np.zeros(steps),
                        'daily': forecast.get('daily', np.zeros(steps)).values if self.daily_seasonality else np.zeros(steps),
                        'yhat_lower': forecast['yhat_lower'].values,
                        'yhat_upper': forecast['yhat_upper'].values
                    }

            except Exception as e:
                print(f"[ERROR] 区域 {region} 预测失败: {str(e)}")
                predictions[region] = np.full(steps, np.nan)

        predictions_df = pd.DataFrame(predictions)

        if return_components:
            return predictions_df, components_dict

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

            common_len = min(len(test_df), len(predictions_df))
            y_true = test_df[region].values[:common_len]
            y_pred = predictions_df[region].values[:common_len]

            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() == 0:
                print(f"[WARNING] 区域 {region} 没有有效的预测数据")
                continue

            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

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

            test_series = test_df[region].iloc[:len(predictions_df)]
            test_time = np.arange(len(train_series), len(train_series) + len(test_series))

            if region in predictions_df.columns:
                pred_series = predictions_df[region].values[:len(test_series)]
                pred_time = test_time

                ax.plot(train_time, train_series.values,
                       label='训练数据', color='#2E86AB', alpha=0.6, linewidth=1.5)
                ax.plot(test_time, test_series.values,
                       label='真实值', color='#06A77D', linewidth=2.5)
                ax.plot(pred_time, pred_series,
                       label='Prophet预测', color='#D62828', linestyle='--', linewidth=2.5)

                # 计算该区域的误差
                mask = ~np.isnan(pred_series)
                if mask.sum() > 0:
                    mae = mean_absolute_error(test_series.values[mask], pred_series[mask])
                    rmse = np.sqrt(mean_squared_error(test_series.values[mask], pred_series[mask]))
                    ax.text(0.02, 0.98, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
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
                  label=f'平均时间: {np.mean(training_times):.2f}秒')
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
                    'seasonality_mode': self.seasonality_mode,
                    'daily_seasonality': self.daily_seasonality,
                    'weekly_seasonality': self.weekly_seasonality,
                    'yearly_seasonality': self.yearly_seasonality,
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'seasonality_prior_scale': self.seasonality_prior_scale,
                    'interval_width': self.interval_width
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
        self.seasonality_mode = config['seasonality_mode']
        self.daily_seasonality = config['daily_seasonality']
        self.weekly_seasonality = config['weekly_seasonality']
        self.yearly_seasonality = config['yearly_seasonality']
        self.changepoint_prior_scale = config['changepoint_prior_scale']
        self.seasonality_prior_scale = config['seasonality_prior_scale']
        self.interval_width = config['interval_width']

        print(f"[OK] 模型加载成功（{len(self.models)} 个区域模型）")


def main():
    """
    主函数：演示Prophet模型训练流程
    """
    print("=" * 80)
    print("Prophet 时间序列预测模型")
    print("=" * 80)

    # 数据路径（支持前端传入配置）
    try:
        from utils.config_loader import load_training_config
        cfg = load_training_config()
        data_path = Path(cfg.get('dataset_path') or (project_root / 'dataset' / 'milano_traffic_nid.csv'))
    except Exception:
        data_path = project_root / 'dataset' / 'milano_traffic_nid.csv'
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)

    # 1. 加载数据
    print(f"\n1. 加载数据: {data_path}")
    df = pd.read_csv(data_path)

    print(f"[OK] 数据加载成功，形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()[:5]}...")

    print("\n2. 划分数据集...")
    try:
        from utils.config_loader import load_training_config
        cfg2 = load_training_config()
        tp = cfg2.get('train_params_by_model', {}).get('Prophet', cfg2.get('train_params', {}))
    except Exception:
        tp = {}
    split_ratio = float((tp or {}).get('train_ratio', 0.9))
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # 选择部分区域进行演示（建议前5个区域测试）
    time_col = df.columns[0]
    all_regions = [col for col in df.columns if col != time_col]

    # 选择前5个区域进行演示
    sample_regions = all_regions[:5]
    print(f"选择区域数: {len(sample_regions)}")
    print(f"训练集长度: {len(train_df)}, 测试集长度: {len(test_df)}")

    train_sample = train_df[[time_col] + sample_regions]
    test_sample = test_df[[time_col] + sample_regions]

    print("\n3. 创建并训练Prophet模型...")
    try:
        mp = cfg2.get('model_params_by_model', {}).get('Prophet', cfg2.get('model_params', {}))
    except Exception:
        mp = {}
    prediction_length = int((tp or {}).get('prediction_length', 288))
    seasonality_mode = str((mp or {}).get('seasonality_mode', 'multiplicative'))
    daily_seasonality = bool((mp or {}).get('daily_seasonality', True))
    weekly_seasonality = bool((mp or {}).get('weekly_seasonality', True))
    yearly_seasonality = bool((mp or {}).get('yearly_seasonality', False))
    changepoint_prior_scale = float((mp or {}).get('changepoint_prior_scale', 0.05))
    seasonality_prior_scale = float((mp or {}).get('seasonality_prior_scale', 10.0))
    n_jobs = int((mp or {}).get('n_jobs', 2))

    forecaster = ProphetForecaster(
        forecast_horizon=prediction_length,
        seasonality_mode=seasonality_mode,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=0.95,
        n_jobs=n_jobs
    )

    # 确定起始时间（从数据中提取）
    if time_col in train_sample.columns:
        start_time = pd.to_datetime(train_sample[time_col].iloc[0])
    else:
        start_time = '2013-11-01 00:00:00'

    train_results = forecaster.fit(
        train_sample,
        time_col,
        sample_regions,
        start_time=start_time,
        parallel=True  # 使用并行训练
    )

    # 4. 预测
    print("\n4. 进行预测...")

    # 确定训练集最后时间点
    last_train_time = start_time + pd.Timedelta(minutes=10 * (len(train_sample) - 1))

    predictions = forecaster.predict(
        sample_regions,
        last_train_time=last_train_time,
        steps=prediction_length
    )

    # 保存预测结果
    pred_path = output_dir / 'prophet_predictions.csv'
    predictions.to_csv(pred_path, index=False)
    print(f"[OK] 预测结果保存到: {pred_path}")

    # 5. 评估
    print("\n5. 评估...")
    test_metrics = forecaster.evaluate(test_sample, predictions, time_col)

    # 保存评估结果
    metrics_path = output_dir / 'prophet_test_metrics.csv'
    test_metrics.to_csv(metrics_path, index=False)
    print(f"[OK] 评估结果保存到: {metrics_path}")

    # 6. 可视化
    print("\n6. 生成可视化...")

    # 预测结果可视化
    pred_plot_path = output_dir / 'prophet_predictions_plot.png'
    forecaster.visualize_predictions(
        train_sample, test_sample, predictions, time_col,
        region_names=sample_regions,  # 显示所有5个区域
        save_path=str(pred_plot_path)
    )

    # 训练汇总可视化
    summary_plot_path = output_dir / 'prophet_training_summary.png'
    forecaster.plot_training_summary(save_path=str(summary_plot_path))

    # 7. 保存模型
    model_path = output_dir / 'prophet_model.pkl'
    forecaster.save_model(str(model_path))

    print("\n" + "=" * 80)
    print("[OK] Prophet模型训练和预测完成！")
    print("=" * 80)
    print(f"\n所有文件已保存到: {output_dir}")
    print(f"  - 预测结果: {pred_path.name}")
    print(f"  - 评估指标: {metrics_path.name}")
    print(f"  - 预测可视化: {pred_plot_path.name}")
    print(f"  - 训练汇总: {summary_plot_path.name}")
    print(f"  - 模型文件: {model_path.name}")


if __name__ == '__main__':
    main()
