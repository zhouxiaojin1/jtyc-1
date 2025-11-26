"""
Chronos-T5（预训练时间序列大模型）

适用：零/少样本、快速获得强基线；支持概率预测与多频率
使用要点：指定频率为 10 分钟、预测步数 1008；可微调以贴合本数据分布
优点：即用即准、泛化强
缺点：模型较大，推理资源占用高
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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

try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("警告: Chronos库未安装。请运行: pip install git+https://github.com/amazon-science/chronos-forecasting.git")


class ChronosForecaster:
    """
    Chronos-T5预训练模型预测器
    """
    def __init__(self,
                 model_name: str = "amazon/chronos-t5-small",
                 prediction_length: int = 1008,
                 num_samples: int = 20,
                 temperature: float = 1.0,
                 top_k: Optional[int] = 50,
                 top_p: Optional[float] = 1.0,
                 device: str = 'auto'):
        """
        Parameters:
        -----------
        model_name : str
            模型名称，可选:
            - amazon/chronos-t5-tiny (8M参数)
            - amazon/chronos-t5-mini (20M参数)
            - amazon/chronos-t5-small (46M参数)
            - amazon/chronos-t5-base (200M参数)
            - amazon/chronos-t5-large (710M参数)
        prediction_length : int
            预测长度
        num_samples : int
            采样数量（用于概率预测）
        temperature : float
            采样温度
        top_k : int
            Top-K采样
        top_p : float
            Top-P采样
        device : str
            设备，'auto'自动检测GPU，'cuda'强制GPU，'cpu'强制CPU
        """
        if not CHRONOS_AVAILABLE:
            raise ImportError("请先安装Chronos库")

        self.model_name = model_name
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # 检测GPU兼容性
        use_gpu = False
        if device == 'auto' or device == 'cuda':
            if torch.cuda.is_available():
                try:
                    # 测试CUDA是否真正可用
                    test_tensor = torch.zeros(1, device='cuda')
                    _ = test_tensor + 1
                    use_gpu = True
                    self.device = 'cuda'
                    print(f"[OK] 检测到GPU: {torch.cuda.get_device_name(0)}")
                    print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                except RuntimeError as e:
                    if 'no kernel image is available' in str(e):
                        print(f"[INFO] GPU检测到但不兼容 (RTX 5060 Ti需要更新的PyTorch)")
                        print("   自动切换到CPU模式")
                        self.device = 'cpu'
                    else:
                        raise
            else:
                self.device = 'cpu'
                print("[INFO] 未检测到GPU，将使用CPU（推理速度会较慢）")
        else:
            self.device = device
            if device == 'cuda':
                print(f"[OK] 使用GPU: {torch.cuda.get_device_name(0)}")

        print(f"\n加载 Chronos 模型: {model_name}")
        print(f"设备: {self.device.upper()}")
        if self.device == 'cuda':
            print("🚀 GPU加速已启用，使用bfloat16精度")

        # 加载预训练模型
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
        )

        print("模型加载完成!")

    def predict_single_series(self, context: np.ndarray,
                             prediction_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        对单个时间序列进行预测

        Parameters:
        -----------
        context : np.ndarray
            历史数据
        prediction_length : int, optional
            预测长度

        Returns:
        --------
        result : dict
            包含 'mean', 'median', 'quantiles' 的预测结果
        """
        if prediction_length is None:
            prediction_length = self.prediction_length

        # 转换为torch张量
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # 预测
        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )

        # 转换为numpy
        forecast_samples = forecast.cpu().numpy()  # (num_samples, prediction_length)

        # 计算统计量
        result = {
            'mean': forecast_samples.mean(axis=0),
            'median': np.median(forecast_samples, axis=0),
            'std': forecast_samples.std(axis=0),
            'q10': np.percentile(forecast_samples, 10, axis=0),
            'q25': np.percentile(forecast_samples, 25, axis=0),
            'q75': np.percentile(forecast_samples, 75, axis=0),
            'q90': np.percentile(forecast_samples, 90, axis=0),
            'samples': forecast_samples
        }

        return result

    def predict(self, df: pd.DataFrame,
                time_col: str,
                value_cols: List[str],
                context_length: int = 2016,
                prediction_length: Optional[int] = None,
                use_median: bool = True) -> pd.DataFrame:
        """
        对多个区域进行预测

        Parameters:
        -----------
        df : pd.DataFrame
            包含历史数据的DataFrame
        time_col : str
            时间列
        value_cols : list
            区域列
        context_length : int
            使用的历史数据长度
        prediction_length : int, optional
            预测长度
        use_median : bool
            是否使用中位数（否则使用均值）

        Returns:
        --------
        predictions_df : pd.DataFrame
            预测结果
        """
        if prediction_length is None:
            prediction_length = self.prediction_length

        print(f"\n开始预测 {len(value_cols)} 个区域，预测步数: {prediction_length}")
        print(f"使用上下文长度: {context_length}")

        predictions = {}
        prediction_intervals = {}

        for region in tqdm(value_cols, desc="预测进度"):
            # 获取历史数据
            context = df[region].values[-context_length:]

            # 处理缺失值
            if np.isnan(context).any():
                context = pd.Series(context).fillna(method='ffill').fillna(method='bfill').values

            # 预测
            try:
                result = self.predict_single_series(context, prediction_length)

                # 选择使用中位数或均值，并确保是1维数组
                if use_median:
                    pred = result['median']
                else:
                    pred = result['mean']

                # 强制展平为1维数组
                predictions[region] = np.array(pred).flatten()

                # 保存置信区间，也确保是1维
                prediction_intervals[region] = {
                    'lower': np.array(result['q10']).flatten(),
                    'upper': np.array(result['q90']).flatten()
                }

            except Exception as e:
                print(f"\n区域 {region} 预测失败: {e}")
                predictions[region] = np.full(prediction_length, np.nan)
                prediction_intervals[region] = {
                    'lower': np.full(prediction_length, np.nan),
                    'upper': np.full(prediction_length, np.nan)
                }

        # 检查所有预测数组的维度和长度
        for region, arr in predictions.items():
            # 确保是1维数组
            predictions[region] = np.array(arr).flatten()
            # 确保置信区间也是1维
            if region in prediction_intervals:
                prediction_intervals[region]['lower'] = np.array(prediction_intervals[region]['lower']).flatten()
                prediction_intervals[region]['upper'] = np.array(prediction_intervals[region]['upper']).flatten()

        lengths = {region: len(arr) for region, arr in predictions.items()}
        if len(set(lengths.values())) > 1:
            print(f"\n[WARN] 警告: 预测结果长度不一致: {lengths}")
            # 统一长度为最小值
            min_length = min(lengths.values())
            predictions = {region: arr[:min_length] for region, arr in predictions.items()}
            prediction_intervals = {
                region: {
                    'lower': intervals['lower'][:min_length],
                    'upper': intervals['upper'][:min_length]
                }
                for region, intervals in prediction_intervals.items()
            }

        predictions_df = pd.DataFrame(predictions)
        self.prediction_intervals = prediction_intervals

        return predictions_df

    def evaluate(self, test_df: pd.DataFrame,
                predictions_df: pd.DataFrame,
                time_col: str) -> pd.DataFrame:
        """
        评估预测结果
        """
        value_cols = [col for col in test_df.columns if col != time_col]

        print(f"\n调试信息:")
        print(f"测试集列: {test_df.columns.tolist()}")
        print(f"预测集列: {predictions_df.columns.tolist()}")
        print(f"待评估的区域列: {value_cols}")
        print(f"测试集长度: {len(test_df)}")
        print(f"预测集长度: {len(predictions_df)}")

        metrics = []
        failed_regions = []
        for region in value_cols:
            if region in predictions_df.columns:
                # 确保长度一致：取两者中的较小值
                min_len = min(len(test_df), len(predictions_df))
                y_true = test_df[region].values[:min_len]
                y_pred = predictions_df[region].values[:min_len]

                # 过滤NaN
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))

                # 检查预测是否全部失败
                if np.isnan(y_pred).all():
                    failed_regions.append(region)
                    continue

                if mask.sum() > 0:
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]

                    mae = mean_absolute_error(y_true_clean, y_pred_clean)
                    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100

                    # 计算覆盖率（真实值在预测区间内的比例）
                    if region in self.prediction_intervals:
                        lower = self.prediction_intervals[region]['lower'][:min_len]
                        upper = self.prediction_intervals[region]['upper'][:min_len]
                        # 只在有效数据点上计算覆盖率
                        coverage = ((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])).mean() * 100
                    else:
                        coverage = np.nan

                    metrics.append({
                        'region': region,
                        'test_mae': mae,
                        'test_rmse': rmse,
                        'test_mape': mape,
                        'coverage_80': coverage
                    })

        metrics_df = pd.DataFrame(metrics)

        print("\n" + "=" * 60)
        print("测试集评估结果:")
        print("=" * 60)

        if len(failed_regions) > 0:
            print(f"\n[WARN] 以下 {len(failed_regions)} 个区域预测失败:")
            for region in failed_regions:
                print(f"   - {region}")

        if len(metrics_df) == 0:
            print("\n警告: 所有区域预测都失败了，没有可评估的数据")
            return metrics_df

        # 检查列是否存在
        if 'test_mae' in metrics_df.columns:
            print(f"平均MAE: {metrics_df['test_mae'].mean():.2f}")
        if 'test_rmse' in metrics_df.columns:
            print(f"平均RMSE: {metrics_df['test_rmse'].mean():.2f}")
        if 'test_mape' in metrics_df.columns:
            print(f"平均MAPE: {metrics_df['test_mape'].mean():.2f}%")
        if 'coverage_80' in metrics_df.columns:
            print(f"80%置信区间覆盖率: {metrics_df['coverage_80'].mean():.2f}%")

        return metrics_df

    def visualize_predictions(self, train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             predictions_df: pd.DataFrame,
                             time_col: str,
                             region_names: Optional[List[str]] = None,
                             n_regions: int = 3,
                             show_intervals: bool = True,
                             save_path: Optional[str] = None):
        """
        可视化预测结果（包含置信区间）
        """
        value_cols = [col for col in train_df.columns if col != time_col]

        if region_names is None:
            region_names = np.random.choice(value_cols, min(n_regions, len(value_cols)), replace=False)

        n_plots = len(region_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5*n_plots))
        if n_plots == 1:
            axes = [axes]

        for idx, region in enumerate(region_names):
            ax = axes[idx]

            # 训练数据（最后一周）
            train_series = train_df[region].iloc[-1008:]
            train_time = np.arange(len(train_series))

            # 测试数据 - 截取到预测长度
            min_len = min(len(test_df), len(predictions_df))
            test_series = test_df[region].iloc[:min_len]
            test_time = np.arange(len(train_series), len(train_series) + len(test_series))

            # 预测数据 - 截取到相同长度
            pred_series = predictions_df[region].values[:min_len]
            pred_time = test_time[:len(pred_series)]

            # 绘制基本曲线
            ax.plot(train_time, train_series.values, label='训练数据', color='blue', alpha=0.7)
            ax.plot(test_time, test_series.values, label='真实值', color='green', linewidth=2)
            ax.plot(pred_time, pred_series, label='预测值 (中位数)', color='red', linestyle='--', linewidth=2)

            # 绘制置信区间
            if show_intervals and region in self.prediction_intervals:
                lower = self.prediction_intervals[region]['lower'][:min_len]
                upper = self.prediction_intervals[region]['upper'][:min_len]

                ax.fill_between(
                    pred_time,
                    lower,
                    upper,
                    color='red',
                    alpha=0.2,
                    label='80% 置信区间'
                )

            ax.set_title(f'区域: {region}', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步 (10分钟间隔)', fontsize=10)
            ax.set_ylabel('交通流量', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=len(train_series), color='black', linestyle=':', linewidth=1, alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n可视化结果已保存到: {save_path}")

        plt.close()

    def visualize_probabilistic_forecast(self, context: np.ndarray,
                                         true_future: Optional[np.ndarray] = None,
                                         region_name: str = "Unknown",
                                         save_path: Optional[str] = None):
        """
        可视化单个区域的概率预测（扇形图）
        """
        # 预测
        result = self.predict_single_series(context)

        fig, ax = plt.subplots(figsize=(15, 6))

        # 确保所有预测结果都是一维数组
        median = np.array(result['median']).flatten()
        q10 = np.array(result['q10']).flatten()
        q90 = np.array(result['q90']).flatten()
        q25 = np.array(result['q25']).flatten()
        q75 = np.array(result['q75']).flatten()

        # 时间轴
        context_time = np.arange(len(context))
        forecast_time = np.arange(len(context), len(context) + len(median))

        # 绘制历史数据
        ax.plot(context_time, context, label='历史数据', color='blue', linewidth=2)

        # 绘制预测中位数
        ax.plot(forecast_time, median, label='预测中位数', color='red', linewidth=2)

        # 绘制置信区间（扇形）
        ax.fill_between(forecast_time, q10, q90,
                       alpha=0.3, color='red', label='10-90%分位数')
        ax.fill_between(forecast_time, q25, q75,
                       alpha=0.5, color='red', label='25-75%分位数')

        # 绘制真实值（如果提供）
        if true_future is not None:
            true_future_flat = np.array(true_future).flatten()
            true_time = forecast_time[:len(true_future_flat)]
            ax.plot(true_time, true_future_flat, label='真实值', color='green',
                   linewidth=2, linestyle='--')

        ax.set_title(f'区域 {region_name} 的概率预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('交通流量', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=len(context), color='black', linestyle=':', linewidth=1, alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n概率预测可视化已保存到: {save_path}")

        plt.close()


def main():
    """
    主函数 - 完整预测流程
    """
    print("=" * 60)
    print("Chronos-T5 预训练时间序列模型")
    print("=" * 60)

    if not CHRONOS_AVAILABLE:
        print("\n错误: Chronos库未安装")
        print("请运行: pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return

    print("\n1. 加载数据...")
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
    print(f"数据形状: {df.shape}")

    tp = cfg.get('train_params_by_model', {}).get('Chronos', cfg.get('train_params', {})) if 'cfg' in locals() else {}
    split_ratio = float(tp.get('train_ratio', 0.9))
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # 选择部分区域
    time_col = df.columns[0]
    all_regions = [col for col in df.columns if col != time_col]
    sample_regions = all_regions[:5]

    train_sample = train_df[[time_col] + sample_regions]
    test_sample = test_df[[time_col] + sample_regions]

    # 3. 创建预测器
    print("\n2. 创建 Chronos 预测器...")
    mp = cfg.get('model_params_by_model', {}).get('Chronos', cfg.get('model_params', {})) if 'cfg' in locals() else {}
    prediction_length = int(tp.get('prediction_length', 288))
    num_samples = int(mp.get('num_samples', 20))
    size_map = {
        'tiny': 'amazon/chronos-t5-tiny',
        'mini': 'amazon/chronos-t5-mini',
        'small': 'amazon/chronos-t5-small',
        'base': 'amazon/chronos-t5-base',
        'large': 'amazon/chronos-t5-large'
    }
    model_size = str(mp.get('model_size', 'small')).lower()
    model_name = size_map.get(model_size, 'amazon/chronos-t5-small')
    temperature = float(mp.get('temperature', 1.0))
    forecaster = ChronosForecaster(
        model_name=model_name,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        device='auto'
    )

    # 创建output目录（使用绝对路径）
    output_dir = os.path.join(script_dir, '..', 'output')
    output_dir = os.path.abspath(output_dir)  # 转换为绝对路径
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 4. 预测
    print("\n3. 进行预测...")
    context_length = int(tp.get('context_length', 2016))
    use_median = bool(mp.get('use_median', True))
    predictions = forecaster.predict(
        train_sample,
        time_col,
        sample_regions,
        context_length=context_length,
        prediction_length=prediction_length,
        use_median=use_median
    )

    # 保存预测结果
    predictions.to_csv(os.path.join(output_dir, 'chronos_predictions.csv'), index=False)
    print("\n预测结果已保存")

    # 5. 评估
    print("\n4. 评估预测结果...")
    test_metrics = forecaster.evaluate(test_sample, predictions, time_col)
    test_metrics.to_csv(os.path.join(output_dir, 'chronos_test_metrics.csv'), index=False)

    # 6. 可视化（带置信区间）
    print("\n5. 生成可视化...")
    forecaster.visualize_predictions(
        train_sample,
        test_sample,
        predictions,
        time_col,
        region_names=sample_regions[:3],
        show_intervals=True,
        save_path=os.path.join(output_dir, 'chronos_predictions_plot.png')
    )

    # 7. 单个区域的概率预测可视化
    print("\n6. 生成概率预测扇形图...")
    sample_region = sample_regions[0]
    context = train_sample[sample_region].values[-2016:]  # 2周历史
    true_future = test_sample[sample_region].values[:288]  # 2天真实值

    forecaster.visualize_probabilistic_forecast(
        context,
        true_future,
        region_name=sample_region,
        save_path=os.path.join(output_dir, 'chronos_probabilistic_plot.png')
    )

    print("\n" + "=" * 60)
    print("[OK] Chronos模型预测完成！")
    print("=" * 60)
    print(f"\n生成的文件 (保存在 {output_dir} 目录):")
    print("  - chronos_predictions.csv: 预测结果")
    print("  - chronos_test_metrics.csv: 测试指标")
    print("  - chronos_predictions_plot.png: 预测可视化（带置信区间）")
    print("  - chronos_probabilistic_plot.png: 概率预测扇形图")
    print("\n注意:")
    print("  - Chronos是预训练模型，无需训练即可使用")
    print("  - 提供概率预测和置信区间")
    print("  - 适合零样本/少样本场景")


if __name__ == "__main__":
    main()
