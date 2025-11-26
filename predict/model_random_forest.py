import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.plot_config import setup_chinese_font, ensure_chinese_font
setup_chinese_font()


class RandomForestForecaster:
    def __init__(self,
                 forecast_horizon: int = 288,
                 lags: Optional[List[int]] = None,
                 rolling_windows: Optional[List[int]] = None,
                 strategy: str = 'recursive',
                 rf_params: Optional[Dict] = None):
        if lags is None:
            lags = [1, 2, 3, 6, 12, 36, 72, 144, 288]
        if rolling_windows is None:
            rolling_windows = [6, 12, 36, 144]
        if rf_params is None:
            rf_params = {
                'n_estimators': 300,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42
            }
        self.forecast_horizon = forecast_horizon
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.strategy = strategy
        self.rf_params = rf_params
        self.models: Dict[str, RandomForestRegressor] = {}
        self.feature_names: List[str] = []
        self.feature_importance: pd.DataFrame = pd.DataFrame()

    def create_time_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        df_time = df.copy()
        df_time['hour'] = df_time[time_col].dt.hour
        df_time['minute'] = df_time[time_col].dt.minute
        df_time['dayofweek'] = df_time[time_col].dt.dayofweek
        df_time['day'] = df_time[time_col].dt.day
        df_time['month'] = df_time[time_col].dt.month
        df_time['is_weekend'] = (df_time['dayofweek'] >= 5).astype(int)
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        df_time['dow_sin'] = np.sin(2 * np.pi * df_time['dayofweek'] / 7)
        df_time['dow_cos'] = np.cos(2 * np.pi * df_time['dayofweek'] / 7)
        return df_time

    def create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df_lag = df.copy()
        for lag in self.lags:
            df_lag[f'lag_{lag}'] = df_lag[target_col].shift(lag)
        return df_lag

    def create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df_roll = df.copy()
        for window in self.rolling_windows:
            df_roll[f'rolling_mean_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).mean()
            df_roll[f'rolling_std_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).std()
            df_roll[f'rolling_min_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).min()
            df_roll[f'rolling_max_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).max()
            df_roll[f'rolling_median_{window}'] = df_roll[target_col].rolling(window=window, min_periods=1).median()
        return df_roll

    def prepare_features(self, df: pd.DataFrame, time_col: str, value_cols: List[str]) -> pd.DataFrame:
        df_long = df.melt(id_vars=[time_col], value_vars=value_cols, var_name='region', value_name='value')
        df_long = self.create_time_features(df_long, time_col)
        dfs = []
        for region in tqdm(value_cols, desc="创建区域特征"):
            df_region = df_long[df_long['region'] == region].copy()
            df_region = self.create_lag_features(df_region, 'value')
            df_region = self.create_rolling_features(df_region, 'value')
            dfs.append(df_region)
        df_features = pd.concat(dfs, axis=0, ignore_index=True)
        df_features['region_id'] = pd.Categorical(df_features['region']).codes
        return df_features

    def fit(self, train_df: pd.DataFrame, time_col: str, value_cols: List[str]) -> Dict:
        train_features = self.prepare_features(train_df, time_col, value_cols)
        max_lag = max(self.lags)
        train_features = train_features.iloc[max_lag:].reset_index(drop=True)
        feature_cols = [c for c in train_features.columns if c not in [time_col, 'region', 'value']]
        self.feature_names = feature_cols
        X_train = train_features[feature_cols]
        y_train = train_features['value']
        model = RandomForestRegressor(**self.rf_params)
        model.fit(X_train, y_train)
        self.models['main'] = model
        y_pred_train = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return {'train_mae': train_mae, 'train_rmse': train_rmse, 'n_features': len(feature_cols), 'strategy': self.strategy}

    def predict(self, test_df: pd.DataFrame, time_col: str, value_cols: List[str], steps: Optional[int] = None) -> pd.DataFrame:
        if steps is None:
            steps = self.forecast_horizon
        model = self.models['main']
        predictions = {region: [] for region in value_cols}
        for region in tqdm(value_cols, desc="递归预测"):
            history = test_df[region].values.copy()
            for step in range(steps):
                features = {}
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
                for lag in self.lags:
                    features[f'lag_{lag}'] = history[-lag] if len(history) >= lag else 0
                for window in self.rolling_windows:
                    if len(history) >= window:
                        window_data = history[-window:]
                        features[f'rolling_mean_{window}'] = np.mean(window_data)
                        features[f'rolling_std_{window}'] = np.std(window_data)
                        features[f'rolling_min_{window}'] = np.min(window_data)
                        features[f'rolling_max_{window}'] = np.max(window_data)
                        features[f'rolling_median_{window}'] = np.median(window_data)
                    else:
                        last = history[-1] if len(history) > 0 else 0
                        features[f'rolling_mean_{window}'] = last
                        features[f'rolling_std_{window}'] = 0
                        features[f'rolling_min_{window}'] = last
                        features[f'rolling_max_{window}'] = last
                        features[f'rolling_median_{window}'] = last
                features['region_id'] = value_cols.index(region)
                X = pd.DataFrame([features])[self.feature_names]
                pred = model.predict(X)[0]
                predictions[region].append(pred)
                history = np.append(history, pred)
        return pd.DataFrame(predictions)

    def evaluate(self, test_df: pd.DataFrame, predictions_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        value_cols = [c for c in test_df.columns if c != time_col]
        metrics = []
        for region in value_cols:
            if region in predictions_df.columns:
                n = min(len(predictions_df), len(test_df))
                y_true = test_df[region].values[:n]
                y_pred = predictions_df[region].values[:n]
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if mask.sum() > 0:
                    yt = y_true[mask]
                    yp = y_pred[mask]
                    mae = mean_absolute_error(yt, yp)
                    rmse = np.sqrt(mean_squared_error(yt, yp))
                    mape = mean_absolute_percentage_error(yt, yp) * 100
                    metrics.append({'region': region, 'test_mae': mae, 'test_rmse': rmse, 'test_mape': mape})
        return pd.DataFrame(metrics)

    def visualize_predictions(self, train_df: pd.DataFrame, test_df: pd.DataFrame, predictions_df: pd.DataFrame,
                              time_col: str, region_names: Optional[List[str]] = None, n_regions: int = 3,
                              save_path: Optional[str] = None):
        ensure_chinese_font()
        value_cols = [c for c in train_df.columns if c != time_col]
        if region_names is None:
            region_names = np.random.choice(value_cols, min(n_regions, len(value_cols)), replace=False)
        n_plots = len(region_names)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5*n_plots))
        if n_plots == 1:
            axes = [axes]
        for idx, region in enumerate(region_names):
            ax = axes[idx]
            train_window = min(1008, len(train_df))
            train_series = train_df[region].iloc[-train_window:]
            train_time = np.arange(len(train_series))
            max_len = min(len(predictions_df), len(test_df))
            test_series = test_df[region].iloc[:max_len]
            test_time = np.arange(len(train_series), len(train_series) + len(test_series))
            pred_series = predictions_df[region].values[:max_len]
            pred_time = test_time
            ax.plot(train_time, train_series.values, label='训练数据', color='#1f77b4', alpha=0.8)
            ax.plot(test_time, test_series.values, label='真实值', color='#2ca02c', linewidth=2)
            ax.plot(pred_time, pred_series, label='预测值', color='#d62728', linestyle='--', linewidth=2)
            ax.set_title(f'区域: {region}', fontsize=12)
            ax.set_xlabel('时间步 (10分钟间隔)')
            ax.set_ylabel('交通流量')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(x=len(train_series), color='black', linestyle=':', linewidth=1, alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('重要性')
        ax.set_title(f'Top {top_n} 特征重要性')
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.models.get('main'),
                'feature_names': self.feature_names,
                'lags': self.lags,
                'rolling_windows': self.rolling_windows
            }, f)


def main():
    from utils.config_loader import load_training_config
    cfg = load_training_config()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = cfg.get('dataset_path')
    if dataset_path is None or not os.path.exists(dataset_path):
        dataset_path = os.path.join(script_dir, '..', 'dataset', 'milano_traffic_nid.csv')
    df = pd.read_csv(dataset_path)
    tp = cfg.get('train_params_by_model', {}).get('RandomForest', cfg.get('train_params', {})) or {}
    mp = cfg.get('model_params_by_model', {}).get('RandomForest', cfg.get('model_params', {})) or {}
    split_ratio = float(tp.get('train_ratio', 0.9))
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    time_col = df.columns[0]
    all_regions = [c for c in df.columns if c != time_col]
    sample_regions = all_regions[:5]
    train_sample = train_df[[time_col] + sample_regions]
    test_sample = test_df[[time_col] + sample_regions]
    prediction_length = int(tp.get('prediction_length', 288))
    rf_params = {
        'n_estimators': int(mp.get('n_estimators', 300)),
        'max_depth': None if mp.get('max_depth', None) in [None, -1] else int(mp.get('max_depth')),
        'min_samples_split': int(mp.get('min_samples_split', 2)),
        'min_samples_leaf': int(mp.get('min_samples_leaf', 1)),
        'max_features': mp.get('max_features', 'sqrt'),
        'n_jobs': int(mp.get('n_jobs', -1)),
        'random_state': 42
    }
    forecaster = RandomForestForecaster(
        forecast_horizon=prediction_length,
        lags=[1, 2, 3, 6, 12, 36, 72, 144, 288],
        rolling_windows=[6, 12, 36, 144],
        strategy='recursive',
        rf_params=rf_params
    )
    train_results = forecaster.fit(train_sample, time_col, sample_regions)
    output_dir = os.path.join(script_dir, '..', 'output')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    predictions = forecaster.predict(train_sample, time_col, sample_regions, steps=prediction_length)
    pd.DataFrame(predictions).to_csv(os.path.join(output_dir, 'randomforest_predictions.csv'), index=False)
    test_metrics = forecaster.evaluate(test_sample, predictions, time_col)
    test_metrics.to_csv(os.path.join(output_dir, 'randomforest_test_metrics.csv'), index=False)
    forecaster.visualize_predictions(train_sample, test_sample, predictions, time_col,
                                     region_names=sample_regions[:3],
                                     save_path=os.path.join(output_dir, 'randomforest_predictions_plot.png'))
    forecaster.plot_feature_importance(top_n=20, save_path=os.path.join(output_dir, 'randomforest_feature_importance.png'))
    forecaster.save_model(os.path.join(output_dir, 'randomforest_model.pkl'))


if __name__ == '__main__':
    main()