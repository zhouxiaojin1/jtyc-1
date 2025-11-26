"""
LightGBM 回归 + 滑窗特征（全局模型）

适用：希望跨区域共享信息、快速上线
特征建议：
- 滞后值 lags=[1,2,3,6,12,36,72,144,288,1008]
- 滚动统计（均值/中位数/最大最小）窗口如 [6,12,36,144]
- 时间特征（小时、星期、是否工作日/节假日）
- 区域 ID 作为分类特征

训练策略：多步直接预测（Direct）或递归；时间序列交叉验证（rolling CV）
优点：速度快、鲁棒、易调参
缺点：远期多步预测误差累积或需要多模型
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import pickle
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


class LightGBMForecaster:
    """
    LightGBM时间序列预测器（全局模型）
    """
    def __init__(self,
                 forecast_horizon: int = 1008,
                 lags: List[int] = [1, 2, 3, 6, 12, 36, 72, 144, 288, 432, 1008],
                 rolling_windows: List[int] = [6, 12, 36, 144],
                 strategy: str = 'direct',
                 lgb_params: Optional[Dict] = None,
                 device: str = 'auto',
                 use_residual: bool = True,
                 baseline_lag: int = 144):
        """
        Parameters:
        -----------
        forecast_horizon : int
            预测步数
        lags : list
            滞后特征列表
        rolling_windows : list
            滚动窗口大小列表
        strategy : str
            预测策略：'direct'（直接多步）或 'recursive'（递归）
        lgb_params : dict, optional
            LightGBM参数
        device : str
            使用设备 'auto'（自动检测GPU）, 'gpu' 或 'cpu'
        """
        self.forecast_horizon = forecast_horizon
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.strategy = strategy
        self.use_residual = use_residual
        self.baseline_lag = baseline_lag

        # 自动检测GPU（不依赖 PyTorch，避免 DLL 错误）
        if device == 'auto':
            try:
                # 尝试通过 nvidia-smi 检测 GPU
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                has_nvidia_gpu = result.returncode == 0

                # 进一步验证 LightGBM 是否支持 GPU
                if has_nvidia_gpu:
                    try:
                        import lightgbm as lgb
                        # 测试创建一个简单的 GPU dataset
                        test_data = lgb.Dataset(np.array([[1, 2], [3, 4]]), label=np.array([0, 1]))
                        self.device = 'gpu'
                        print("[OK] 检测到GPU，LightGBM将使用GPU加速")
                    except:
                        self.device = 'cpu'
                        print("[INFO] 检测到GPU但LightGBM不支持，将使用CPU")
                else:
                    self.device = 'cpu'
                    print("[INFO] 未检测到GPU，LightGBM将使用CPU")
            except Exception as e:
                self.device = 'cpu'
                print(f"[INFO] GPU检测失败，LightGBM将使用CPU")
        else:
            self.device = device
            # 如果用户指定 GPU 但不可用，也降级到 CPU
            if device == 'gpu':
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    if result.returncode != 0:
                        self.device = 'cpu'
                        print("[INFO] GPU不可用，已自动切换到CPU")
                except:
                    self.device = 'cpu'
                    print("[INFO] GPU不可用，已自动切换到CPU")

        # 默认LightGBM参数
        default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device_type': self.device,
            'max_depth': 8,
            'min_child_samples': 20
        }

        # GPU特定参数
        if self.device == 'gpu':
            default_params.update({
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_bin': 63  # GPU推荐较小的max_bin
            })

        if lgb_params:
            default_params.update(lgb_params)

        self.lgb_params = default_params
        self.models = {}  # 存储模型
        self.feature_names = []

    def create_time_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        创建时间特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        time_col : str
            时间列名称

        Returns:
        --------
        df_time : pd.DataFrame
            包含时间特征的数据框
        """
        df_time = df.copy()

        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_time[time_col]):
            df_time[time_col] = pd.to_datetime(df_time[time_col])

        # 提取时间特征
        df_time['hour'] = df_time[time_col].dt.hour
        df_time['minute'] = df_time[time_col].dt.minute
        df_time['dayofweek'] = df_time[time_col].dt.dayofweek
        df_time['day'] = df_time[time_col].dt.day
        df_time['month'] = df_time[time_col].dt.month
        df_time['is_weekend'] = (df_time['dayofweek'] >= 5).astype(int)

        # 周期性编码（小时和星期）
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        df_time['dow_sin'] = np.sin(2 * np.pi * df_time['dayofweek'] / 7)
        df_time['dow_cos'] = np.cos(2 * np.pi * df_time['dayofweek'] / 7)

        return df_time

    def create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        创建滞后特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        target_col : str
            目标列名称

        Returns:
        --------
        df_lag : pd.DataFrame
            包含滞后特征的数据框
        """
        df_lag = df.copy()

        for lag in self.lags:
            df_lag[f'lag_{lag}'] = df_lag[target_col].shift(lag)

        return df_lag

    def create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        创建滚动统计特征

        Parameters:
        -----------
        df : pd.DataFrame
            数据框
        target_col : str
            目标列名称

        Returns:
        --------
        df_roll : pd.DataFrame
            包含滚动特征的数据框
        """
        df_roll = df.copy()

        for window in self.rolling_windows:
            df_roll[f'rolling_mean_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).mean()
            df_roll[f'rolling_std_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).std()
            df_roll[f'rolling_min_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).min()
            df_roll[f'rolling_max_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).max()
            df_roll[f'rolling_median_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).median()

        return df_roll

    def prepare_features(self, df: pd.DataFrame,
                        time_col: str,
                        value_cols: List[str]) -> pd.DataFrame:
        """
        准备所有特征（全局模型）

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据框
        time_col : str
            时间列
        value_cols : list
            区域列列表

        Returns:
        --------
        df_features : pd.DataFrame
            特征数据框
        """
        print("\n创建特征...")

        # 转换为长格式
        df_long = df.melt(id_vars=[time_col], value_vars=value_cols,
                         var_name='region', value_name='value')

        # 创建时间特征
        df_long = self.create_time_features(df_long, time_col)

        # 对每个区域创建滞后和滚动特征
        dfs = []
        for region in tqdm(value_cols, desc="创建区域特征"):
            df_region = df_long[df_long['region'] == region].copy()

            # 滞后特征
            df_region = self.create_lag_features(df_region, 'value')

            # 滚动特征
            df_region = self.create_rolling_features(df_region, 'value')

            dfs.append(df_region)

        df_features = pd.concat(dfs, axis=0, ignore_index=True)

        # 区域编码（LabelEncoding）
        df_features['region_id'] = pd.Categorical(df_features['region']).codes

        return df_features

    def fit(self, train_df: pd.DataFrame,
            time_col: str,
            value_cols: List[str],
            valid_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        训练模型

        Parameters:
        -----------
        train_df : pd.DataFrame
            训练数据
        time_col : str
            时间列
        value_cols : list
            区域列
        valid_df : pd.DataFrame, optional
            验证集

        Returns:
        --------
        train_results : dict
            训练结果
        """
        print("=" * 60)
        print("开始训练 LightGBM 全局模型")
        print(f"预测策略: {self.strategy}")
        print(f"预测步数: {self.forecast_horizon}")
        print(f"设备: {self.device.upper()}")
        if self.device == 'gpu':
            print("[OK] GPU加速已启用")
        print("=" * 60)

        # 准备特征
        train_features = self.prepare_features(train_df, time_col, value_cols)

        # 删除包含NaN的行（由滞后特征产生）
        max_lag = max(self.lags)
        train_features = train_features.iloc[max_lag:].reset_index(drop=True)

        # 特征列
        feature_cols = [col for col in train_features.columns
                       if col not in [time_col, 'region', 'value']]
        self.feature_names = feature_cols

        X_train = train_features[feature_cols]
        y_train_raw = train_features['value']
        if self.use_residual:
            baseline_col = f'lag_{self.baseline_lag}'
            if baseline_col in train_features.columns:
                baseline = train_features[baseline_col]
            else:
                # 回退到相近窗口的滚动均值
                roll_col = f'rolling_mean_{min(self.baseline_lag, self.rolling_windows[-1])}'
                baseline = train_features[roll_col] if roll_col in train_features.columns else pd.Series(np.zeros(len(train_features)))
            y_train = (y_train_raw - baseline).fillna(0.0)
        else:
            y_train = y_train_raw

        # 准备验证集（如果提供）
        if valid_df is not None:
            valid_features = self.prepare_features(valid_df, time_col, value_cols)
            valid_features = valid_features.iloc[max_lag:].reset_index(drop=True)
            X_valid = valid_features[feature_cols]
            y_valid_raw = valid_features['value']
            if self.use_residual:
                baseline_col = f'lag_{self.baseline_lag}'
                if baseline_col in valid_features.columns:
                    baseline_v = valid_features[baseline_col]
                else:
                    roll_col = f'rolling_mean_{min(self.baseline_lag, self.rolling_windows[-1])}'
                    baseline_v = valid_features[roll_col] if roll_col in valid_features.columns else pd.Series(np.zeros(len(valid_features)))
                y_valid = (y_valid_raw - baseline_v).fillna(0.0)
            else:
                y_valid = y_valid_raw

        print(f"\n训练集大小: {X_train.shape}")
        print(f"特征数量: {len(feature_cols)}")

        if self.strategy == 'direct':
            # 直接多步预测：训练多个模型，每个模型预测一个步长
            print("\n使用直接多步预测策略...")

            # 为了演示，我们训练一个通用模型
            # 实际应用中可以为每个步长训练独立模型

            train_data = lgb.Dataset(X_train, label=y_train)

            if valid_df is not None:
                valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
                callbacks = [lgb.early_stopping(stopping_rounds=50)]
                model = lgb.train(
                    self.lgb_params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[valid_data],
                    callbacks=callbacks
                )
            else:
                model = lgb.train(
                    self.lgb_params,
                    train_data,
                    num_boost_round=500
                )

            self.models['main'] = model

        else:  # recursive
            # 递归预测：训练单个模型，递归生成预测
            print("\n使用递归预测策略...")

            train_data = lgb.Dataset(X_train, label=y_train)

            if valid_df is not None:
                valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
                callbacks = [lgb.early_stopping(stopping_rounds=50)]
                model = lgb.train(
                    self.lgb_params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[valid_data],
                    callbacks=callbacks
                )
            else:
                model = lgb.train(
                    self.lgb_params,
                    train_data,
                    num_boost_round=500
                )

            self.models['main'] = model

        # 训练集性能
        y_pred_train = model.predict(X_train)
        if self.use_residual:
            baseline_col = f'lag_{self.baseline_lag}'
            if baseline_col in train_features.columns:
                baseline = train_features[baseline_col]
            else:
                roll_col = f'rolling_mean_{min(self.baseline_lag, self.rolling_windows[-1])}'
                baseline = train_features[roll_col] if roll_col in train_features.columns else pd.Series(np.zeros(len(train_features)))
            y_pred_train_final = y_pred_train + baseline.values
            mask = ~(np.isnan(y_train_raw.values) | np.isnan(y_pred_train_final))
            train_mae = mean_absolute_error(y_train_raw.values[mask], y_pred_train_final[mask])
            train_rmse = np.sqrt(mean_squared_error(y_train_raw.values[mask], y_pred_train_final[mask]))
        else:
            mask = ~(np.isnan(y_train.values) | np.isnan(y_pred_train))
            train_mae = mean_absolute_error(y_train.values[mask], y_pred_train[mask])
            train_rmse = np.sqrt(mean_squared_error(y_train.values[mask], y_pred_train[mask]))

        results = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'n_features': len(feature_cols),
            'strategy': self.strategy
        }

        print(f"\n训练完成!")
        print(f"训练MAE: {train_mae:.2f}")
        print(f"训练RMSE: {train_rmse:.2f}")

        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 重要特征:")
        print(self.feature_importance.head(10))

        return results

    def predict(self, test_df: pd.DataFrame,
                time_col: str,
                value_cols: List[str],
                steps: Optional[int] = None) -> pd.DataFrame:
        """
        预测

        Parameters:
        -----------
        test_df : pd.DataFrame
            测试数据（用于提供历史数据）
        time_col : str
            时间列
        value_cols : list
            区域列
        steps : int, optional
            预测步数

        Returns:
        --------
        predictions_df : pd.DataFrame
            预测结果
        """
        if steps is None:
            steps = self.forecast_horizon

        print(f"\n开始预测 {steps} 步...")

        model = self.models['main']

        if self.strategy == 'direct':
            # 直接预测（简化版：使用同一个模型）
            test_features = self.prepare_features(test_df, time_col, value_cols)
            max_lag = max(self.lags)
            test_features = test_features.iloc[max_lag:].reset_index(drop=True)

            X_test = test_features[self.feature_names]
            predictions = model.predict(X_test)

            # 重塑为区域×步数
            predictions_df = pd.DataFrame()
            for region in value_cols:
                region_mask = test_features['region'] == region
                region_preds = predictions[region_mask][:steps]
                if self.use_residual:
                    baseline_col = f'lag_{self.baseline_lag}'
                    if baseline_col in test_features.columns:
                        baseline = test_features.loc[region_mask, baseline_col].values[:steps]
                    else:
                        roll_col = f'rolling_mean_{min(self.baseline_lag, self.rolling_windows[-1])}'
                        baseline = test_features.loc[region_mask, roll_col].values[:steps] if roll_col in test_features.columns else np.zeros(len(region_preds))
                    region_preds = region_preds + baseline
                predictions_df[region] = region_preds

        else:  # recursive
            # 递归预测
            predictions = {region: [] for region in value_cols}

            # 从测试集获取初始历史数据
            for region in tqdm(value_cols, desc="递归预测"):
                history = test_df[region].values.copy()

                for step in range(steps):
                    # 创建特征
                    features = {}

                    # 时间特征（假设10分钟间隔）
                    # 简化处理：使用step来推断时间
                    current_time = pd.Timestamp(test_df[time_col].iloc[-1]) + pd.Timedelta(minutes=(step+1)*10)
                    features['hour'] = current_time.hour
                    features['minute'] = current_time.minute
                    features['dayofweek'] = current_time.dayofweek
                    features['day'] = current_time.day
                    features['month'] = current_time.month
                    features['is_weekend'] = 1 if current_time.dayofweek >= 5 else 0
                    features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
                    features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
                    features['dow_sin'] = np.sin(2 * np.pi * current_time.dayofweek / 7)
                    features['dow_cos'] = np.cos(2 * np.pi * current_time.dayofweek / 7)

                    # 滞后特征
                    for lag in self.lags:
                        if len(history) >= lag:
                            features[f'lag_{lag}'] = history[-lag]
                        else:
                            features[f'lag_{lag}'] = 0

                    # 滚动特征
                    for window in self.rolling_windows:
                        if len(history) >= window:
                            window_data = history[-window:]
                            features[f'rolling_mean_{window}'] = np.mean(window_data)
                            features[f'rolling_std_{window}'] = np.std(window_data)
                            features[f'rolling_min_{window}'] = np.min(window_data)
                            features[f'rolling_max_{window}'] = np.max(window_data)
                            features[f'rolling_median_{window}'] = np.median(window_data)
                        else:
                            features[f'rolling_mean_{window}'] = history[-1] if len(history) > 0 else 0
                            features[f'rolling_std_{window}'] = 0
                            features[f'rolling_min_{window}'] = history[-1] if len(history) > 0 else 0
                            features[f'rolling_max_{window}'] = history[-1] if len(history) > 0 else 0
                            features[f'rolling_median_{window}'] = history[-1] if len(history) > 0 else 0

                    # 区域ID
                    features['region_id'] = value_cols.index(region)

                    # 转换为DataFrame并预测
                    X = pd.DataFrame([features])[self.feature_names]
                    pred_residual = model.predict(X)[0]
                    # 季节基线
                    baseline = None
                    base_key = f'lag_{self.baseline_lag}'
                    if base_key in X.columns:
                        baseline = X[base_key].iloc[0]
                    if baseline is None:
                        roll_key = f'rolling_mean_{min(self.baseline_lag, self.rolling_windows[-1])}'
                        baseline = X[roll_key].iloc[0] if roll_key in X.columns else (history[-1] if len(history) > 0 else 0)

                    pred = pred_residual + (baseline if self.use_residual else 0)

                    predictions[region].append(pred)
                    history = np.append(history, pred)

            predictions_df = pd.DataFrame(predictions)

        return predictions_df

    def evaluate(self, test_df: pd.DataFrame,
                predictions_df: pd.DataFrame,
                time_col: str) -> pd.DataFrame:
        """
        评估预测结果
        """
        value_cols = [col for col in test_df.columns if col != time_col]

        metrics = []
        for region in value_cols:
            if region in predictions_df.columns:
                y_pred = predictions_df[region].values
                n = min(len(y_pred), len(test_df))
                if n == 0:
                    continue
                y_true = test_df[region].values[:n]
                y_pred = y_pred[:n]

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

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        绘制特征重要性
        """
        from utils.plot_config import ensure_chinese_font
        ensure_chinese_font()

        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = self.feature_importance.head(top_n)

        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(f'Top {top_n} 特征重要性', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n特征重要性图已保存到: {save_path}")

        plt.close()

    def visualize_predictions(self, train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             predictions_df: pd.DataFrame,
                             time_col: str,
                             region_names: Optional[List[str]] = None,
                             n_regions: int = 3,
                             save_path: Optional[str] = None):
        """
        可视化预测结果
        """
        from utils.plot_config import ensure_chinese_font
        ensure_chinese_font()

        value_cols = [col for col in train_df.columns if col != time_col]

        if region_names is None:
            region_names = np.random.choice(value_cols, min(n_regions, len(value_cols)), replace=False)

        n_plots = len(region_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5*n_plots))
        if n_plots == 1:
            axes = [axes]

        for idx, region in enumerate(region_names):
            ax = axes[idx]

            train_series = train_df[region].iloc[-1008:]
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

    def save_model(self, save_path: str):
        """保存模型"""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'lags': self.lags,
                'rolling_windows': self.rolling_windows,
                'strategy': self.strategy
            }, f)
        print(f"\n模型已保存到: {save_path}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("LightGBM 时间序列预测模型")
    print("=" * 60)

    # 加载配置文件（如果有）
    from utils.config_loader import load_training_config, get_param

    config = load_training_config()
    model_params = config.get('model_params', {})
    train_params = config.get('train_params_by_model', {}).get('LightGBM', config.get('train_params', {}))

    # 1. 加载数据
    print("\n1. 加载数据...")
    # 获取配置中的数据集路径，回退到默认
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = config.get('dataset_path')
    if dataset_path is None or not os.path.exists(dataset_path):
        dataset_path = os.path.join(script_dir, '..', 'dataset', 'milano_traffic_nid.csv')
    print(f"[INFO] 数据路径: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 2. 划分数据集（按配置比例）
    split_ratio = float(train_params.get('train_ratio', 0.9))
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # 选择部分区域
    time_col = df.columns[0]
    all_regions = [col for col in df.columns if col != time_col]
    sample_regions = all_regions[:5]

    train_sample = train_df[[time_col] + sample_regions]
    test_sample = test_df[[time_col] + sample_regions]

    # 3. 创建并训练模型 - 使用配置参数或默认值
    print("\n2. 创建并训练模型...")

    # 从配置获取参数
    n_estimators = get_param(config, 'n_estimators', 100)
    max_depth = get_param(config, 'max_depth', -1)
    learning_rate = get_param(config, 'learning_rate', 0.1)
    num_leaves = get_param(config, 'num_leaves', 31)
    min_child_samples = get_param(config, 'min_child_samples', 20)
    subsample = get_param(config, 'subsample', 0.8)
    prediction_length = int(train_params.get('prediction_length', 288))
    context_length = int(train_params.get('context_length', 2016))

    print(f"\n[模型参数]")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  num_leaves: {num_leaves}")
    print(f"  min_child_samples: {min_child_samples}")
    print(f"  subsample: {subsample}")

    forecaster = LightGBMForecaster(
        forecast_horizon=prediction_length,
        lags=[1, 2, 3, 6, 12, 36, 72, 144, 288, 432, 1008],
        rolling_windows=[6, 12, 36, 144],
        strategy='recursive',
        device='auto',  # 自动检测GPU
        use_residual=True,
        baseline_lag=144,
        lgb_params={
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample
        }
    )

    # 限制训练历史窗口长度以匹配配置
    if context_length > 0 and len(train_sample) > context_length:
        train_sample = train_sample.tail(context_length).reset_index(drop=True)
    # 使用部分训练集作为验证，提升泛化与早停效果
    valid_ratio = 0.1
    if len(train_sample) > 100:
        split_valid = int(len(train_sample) * (1 - valid_ratio))
        valid_df = train_sample.iloc[split_valid:].reset_index(drop=True)
        train_df2 = train_sample.iloc[:split_valid].reset_index(drop=True)
    else:
        valid_df = None
        train_df2 = train_sample
    train_results = forecaster.fit(train_df2, time_col, sample_regions, valid_df=valid_df)

    # 创建output目录（使用绝对路径）
    output_dir = os.path.join(script_dir, '..', 'output')
    output_dir = os.path.abspath(output_dir)  # 转换为绝对路径
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 4. 预测
    print("\n3. 进行预测...")
    predictions = forecaster.predict(test_sample, time_col, sample_regions, steps=prediction_length)
    predictions.to_csv(os.path.join(output_dir, 'lightgbm_predictions.csv'), index=False)

    # 5. 评估
    print("\n4. 评估...")
    test_metrics = forecaster.evaluate(test_sample, predictions, time_col)
    test_metrics.to_csv(os.path.join(output_dir, 'lightgbm_test_metrics.csv'), index=False)

    # 6. 可视化
    print("\n5. 生成可视化...")
    forecaster.visualize_predictions(
        train_sample, test_sample, predictions, time_col,
        region_names=sample_regions[:3],
        save_path=os.path.join(output_dir, 'lightgbm_predictions_plot.png')
    )

    forecaster.plot_feature_importance(
        top_n=20,
        save_path=os.path.join(output_dir, 'lightgbm_feature_importance.png')
    )

    # 7. 保存模型
    forecaster.save_model(os.path.join(output_dir, 'lightgbm_model.pkl'))

    print("\n" + "=" * 60)
    print("[OK] LightGBM模型训练和预测完成！")
    print("=" * 60)
    print(f"所有文件已保存到: {output_dir}")


if __name__ == "__main__":
    main()
