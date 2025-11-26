"""
结果分析页面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入字体配置
from utils.plot_config import setup_chinese_font, apply_plot_style

# 配置中文字体
setup_chinese_font()


def show():
    """显示结果分析页面"""
    st.title("结果分析")

    # 检查是否有输出文件
    output_path = Path("output")
    if not output_path.exists():
        st.warning("输出目录不存在，请先训练模型")
        return

    # 查找预测结果文件（排除 TBATS）
    prediction_files = [
        f for f in output_path.glob("*_predictions.csv")
        if "tbats" not in f.name.lower() and "lstm" not in f.name.lower() and "randomforest" not in f.name.lower()
    ]
    metrics_files = [
        f for f in output_path.glob("*_test_metrics.csv")
        if "tbats" not in f.name.lower() and "lstm" not in f.name.lower() and "randomforest" not in f.name.lower()
    ]

    if not prediction_files:
        st.info("暂无模型预测结果，请先训练模型")
        return

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["预测结果", "模型对比", "详细分析"])

    with tab1:
        show_prediction_results(prediction_files, metrics_files, output_path)

    with tab2:
        show_model_comparison(metrics_files, output_path)

    with tab3:
        show_detailed_analysis(prediction_files, metrics_files, output_path)


def show_prediction_results(prediction_files, metrics_files, output_path):
    """显示预测结果"""
    st.markdown("### 预测结果")

    # 选择模型
    model_names = [f.stem.replace("_predictions", "") for f in prediction_files]
    selected_model = st.selectbox("选择模型", model_names)

    # 展示当前模型对应的数据集与训练参数提示
    cfg_path = Path("config/training_config.json")
    if cfg_path.exists():
        try:
            import json
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            history = cfg.get('history', {})
            ds_by_model = history.get('dataset_path_by_model', {})
            ds_path = ds_by_model.get(selected_model, cfg.get('dataset_path'))
            tp_by_model = history.get('train_params_by_model', {})
            tp = tp_by_model.get(selected_model, cfg.get('train_params', {}))
            pr = tp.get('prediction_length')
            tr = tp.get('train_ratio')
            cl = tp.get('context_length')
            info_text = f"数据集：{Path(ds_path).name if ds_path else '未配置'} | 训练比例：{tr if tr is not None else 0.9} | 历史窗口：{cl if cl is not None else '默认'} | 预测步数：{pr if pr is not None else '默认'}"
            st.info(info_text)
        except Exception:
            pass

    # 找到对应的文件
    pred_file = output_path / f"{selected_model}_predictions.csv"
    metrics_file = output_path / f"{selected_model}_test_metrics.csv"

    if not pred_file.exists():
        st.error(f"❌ 找不到预测文件：{pred_file}")
        return

    # 加载预测结果
    try:
        predictions = pd.read_csv(pred_file)

        st.markdown("#### 预测数据")
        st.dataframe(predictions.head(100), use_container_width=True, height=300)

        # 基本统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("预测步数", len(predictions))

        with col2:
            st.metric("区域数量", len(predictions.columns))

        with col3:
            mean_pred = predictions.mean().mean()
            st.metric("平均预测值", f"{mean_pred:.2f}")

        with col4:
            std_pred = predictions.std().mean()
            st.metric("平均标准差", f"{std_pred:.2f}")

        # 加载评估指标
        if metrics_file.exists():
            st.markdown("---")
            st.markdown("#### 评估指标")

            metrics = pd.read_csv(metrics_file)

            # 显示指标表格
            st.dataframe(metrics, use_container_width=True)

            # 指标摘要
            if len(metrics) > 0:
                col1, col2, col3 = st.columns(3)

                with col1:
                    if 'test_mae' in metrics.columns:
                        avg_mae = metrics['test_mae'].mean()
                        st.metric("平均 MAE", f"{avg_mae:.2f}")

                with col2:
                    if 'test_rmse' in metrics.columns:
                        avg_rmse = metrics['test_rmse'].mean()
                        st.metric("平均 RMSE", f"{avg_rmse:.2f}")

                with col3:
                    if 'test_mape' in metrics.columns:
                        avg_mape = metrics['test_mape'].mean()
                        st.metric("平均 MAPE", f"{avg_mape:.2f}%")

                # 可视化指标分布
                st.markdown("#### 指标分布")

                metric_cols = [c for c in metrics.columns if c != 'region']

                if len(metric_cols) > 0:
                    selected_metric = st.selectbox("选择指标", metric_cols)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(metrics[selected_metric].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
                    apply_plot_style(ax,
                                   title=f'{selected_metric} 分布',
                                   xlabel=selected_metric,
                                   ylabel='频数',
                                   grid=True,
                                   legend=False)
                    st.pyplot(fig)
                    plt.close()

        # 可视化预测结果
        st.markdown("---")
        st.markdown("#### 预测可视化")

        is_pair_format = 'y_pred' in predictions.columns and 'y_true' in predictions.columns

        # 如果是y_true/y_pred格式（LSTM等深度学习模型）
        if is_pair_format:
            st.info("检测到深度学习模型输出格式（y_true vs y_pred）")

            # 绘制预测值vs真实值对比图
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 第一张图：预测值和真实值对比（折线图）
            ax = axes[0]
            time_steps = np.arange(len(predictions))

            ax.plot(time_steps, predictions['y_true'].values,
                   linewidth=2, label='真实值', color='#2ca02c', alpha=0.9)
            ax.plot(time_steps, predictions['y_pred'].values,
                   linewidth=2, label='预测值', color='#d62728', linestyle='--', alpha=0.9)

            # 计算误差
            mae = np.mean(np.abs(predictions['y_true'].values - predictions['y_pred'].values))
            rmse = np.sqrt(np.mean((predictions['y_true'].values - predictions['y_pred'].values)**2))

            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            apply_plot_style(ax,
                           title='预测值 vs 真实值对比',
                           xlabel='时间步',
                           ylabel='标准化后的值',
                           grid=True,
                           legend=True)

            # 第二张图：误差分布图
            ax = axes[1]
            errors = predictions['y_pred'].values - predictions['y_true'].values

            ax.plot(time_steps, errors, linewidth=1.5, color='#ff7f0e', alpha=0.7, label='预测误差')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.fill_between(time_steps, 0, errors, alpha=0.3, color='#ff7f0e')

            apply_plot_style(ax,
                           title='预测误差随时间变化',
                           xlabel='时间步',
                           ylabel='误差 (预测值 - 真实值)',
                           grid=True,
                           legend=True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 添加散点图对比
            st.markdown("##### 散点图分析")
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.scatter(predictions['y_true'].values, predictions['y_pred'].values,
                      alpha=0.5, s=20, color='#1f77b4')

            # 添加理想预测线（y=x）
            min_val = min(predictions['y_true'].min(), predictions['y_pred'].min())
            max_val = max(predictions['y_true'].max(), predictions['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'r--', linewidth=2, label='理想预测线 (y=x)')

            # 计算相关系数
            corr = np.corrcoef(predictions['y_true'].values, predictions['y_pred'].values)[0, 1]
            ax.text(0.05, 0.95, f'相关系数: {corr:.4f}',
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            apply_plot_style(ax,
                           title='真实值 vs 预测值散点图',
                           xlabel='真实值',
                           ylabel='预测值',
                           grid=True,
                           legend=True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 显示统计信息
            st.markdown("##### 预测统计")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("平均绝对误差 (MAE)", f"{mae:.4f}")

            with col2:
                st.metric("均方根误差 (RMSE)", f"{rmse:.4f}")

            with col3:
                st.metric("相关系数", f"{corr:.4f}")

            with col4:
                mape = np.mean(np.abs((predictions['y_true'].values - predictions['y_pred'].values) /
                                     (predictions['y_true'].values + 1e-8))) * 100
                st.metric("平均百分比误差 (MAPE)", f"{mape:.2f}%")

            return  # 对于pair格式，到这里就结束

        # 以下是原来的多区域格式处理
        # 加载原始数据以获取历史值和真实值
        # 优先使用训练配置中的数据集路径
        cfg_path = Path("config/training_config.json")
        data_path = None
        train_ratio = 0.9
        if cfg_path.exists():
            try:
                import json
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                history = cfg.get('history', {})
                ds_by_model = history.get('dataset_path_by_model', {})
                dp_global = cfg.get('dataset_path')
                dp = ds_by_model.get(selected_model, dp_global)
                if dp:
                    data_path = Path(dp)
                tp_by_model = history.get('train_params_by_model', {})
                trp = tp_by_model.get(selected_model, cfg.get('train_params', {})).get('train_ratio')
                if isinstance(trp, (int, float)):
                    train_ratio = float(trp)
            except Exception:
                pass
        if data_path is None or not data_path.exists():
            data_path = Path("dataset/milano_traffic_nid.csv")
            if not data_path.exists():
                data_path = Path("dataset/trentino_traffic_nid.csv")

        has_comparison_data = False
        train_df = None
        test_df = None

        if data_path.exists():
            try:
                # 加载完整数据
                full_data = pd.read_csv(data_path)

                # 使用配置中的训练比例（默认0.9）
                split_idx = int(len(full_data) * train_ratio)
                train_df = full_data.iloc[:split_idx]
                test_df = full_data.iloc[split_idx:split_idx + len(predictions)]

                # 检查是否有时间列
                time_col = full_data.columns[0]
                has_comparison_data = True

            except Exception as e:
                st.warning(f"无法加载原始数据进行对比: {str(e)}")

        # 选择区域
        regions = predictions.columns.tolist()
        selected_regions = st.multiselect(
            "选择要可视化的区域（最多5个）",
            regions,
            default=(regions[:3] if len(regions) >= 3 else regions)
        )

        if len(selected_regions) > 5:
            st.warning("⚠️ 最多选择5个区域")
            selected_regions = selected_regions[:5]

        if selected_regions:
            if has_comparison_data and train_df is not None and test_df is not None:
                # 绘制完整的对比图：训练数据 + 真实值 + 预测值
                fig, axes = plt.subplots(len(selected_regions), 1,
                                        figsize=(14, 5*len(selected_regions)))

                if len(selected_regions) == 1:
                    axes = [axes]

                for idx, region in enumerate(selected_regions):
                    ax = axes[idx]

                    # 获取预测数据长度
                    pred_len = len(predictions)

                    # 获取训练数据（显示最后1008个点，约7天）
                    train_window = min(1008, len(train_df))
                    train_data = train_df[region].iloc[-train_window:].values
                    train_time = np.arange(len(train_data))

                    # 获取测试数据（真实值），长度与预测数据匹配
                    test_data_available = min(pred_len, len(test_df))
                    test_data = test_df[region].values[:test_data_available]

                    # 获取预测数据（可能需要截断）
                    pred_data = predictions[region].values[:test_data_available]

                    # 计算时间轴（确保长度一致）
                    test_time = np.arange(len(train_data), len(train_data) + test_data_available)
                    pred_time = test_time[:len(pred_data)]

                    # 绘制训练数据（蓝色）
                    ax.plot(train_time, train_data, linewidth=1.5,
                           label='训练数据', color='#1f77b4', alpha=0.8)

                    # 绘制真实值（绿色） - 只绘制与预测长度相同的部分
                    if len(test_data) > 0:
                        ax.plot(test_time[:len(test_data)], test_data, linewidth=2,
                               label='真实值', color='#2ca02c', alpha=0.9)

                    # 绘制预测值（红色虚线）
                    ax.plot(pred_time, pred_data, linewidth=2,
                           label='预测值', color='#d62728', linestyle='--', alpha=0.9)

                    # 绘制分界线
                    ax.axvline(x=len(train_data), color='gray',
                              linestyle=':', linewidth=1.5, alpha=0.7)

                    # 设置标题和标签
                    apply_plot_style(ax,
                                   title=f'区域: {region}',
                                   xlabel='时间步 (10分钟间隔)',
                                   ylabel='交通流量',
                                   grid=True,
                                   legend=True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            else:
                # 仅绘制预测曲线（无历史数据对比）
                fig, axes = plt.subplots(len(selected_regions), 1,
                                        figsize=(12, 4*len(selected_regions)))

                if len(selected_regions) == 1:
                    axes = [axes]

                for idx, region in enumerate(selected_regions):
                    ax = axes[idx]
                    ax.plot(predictions[region].values, linewidth=2,
                           label='预测值', color='#d62728')

                    apply_plot_style(ax,
                                   title=f'区域: {region}',
                                   xlabel='时间步',
                                   ylabel='交通流量',
                                   grid=True,
                                   legend=True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.info("提示：放置原始数据文件到 dataset 目录可查看完整的历史值和真实值对比")

    except Exception as e:
        st.error(f"加载预测结果失败：{str(e)}")


def show_model_comparison(metrics_files, output_path):
    """显示模型对比"""
    st.markdown("### 模型对比")

    if len(metrics_files) < 2:
        st.info("需要至少 2 个模型才能进行对比")
        return

    # 加载所有模型的指标
    all_metrics = {}

    for metrics_file in metrics_files:
        model_name = metrics_file.stem.replace("_test_metrics", "")
        try:
            metrics = pd.read_csv(metrics_file)
            all_metrics[model_name] = metrics
        except Exception as e:
            st.warning(f"无法加载 {model_name} 的指标：{str(e)}")

    if len(all_metrics) == 0:
        st.error("无法加载任何模型指标")
        return

    # 计算平均指标
    st.markdown("#### 平均性能对比")

    comparison_data = []

    for model_name, metrics in all_metrics.items():
        row = {'模型': model_name}

        if 'test_mae' in metrics.columns:
            row['MAE'] = metrics['test_mae'].mean()

        if 'test_rmse' in metrics.columns:
            row['RMSE'] = metrics['test_rmse'].mean()

        if 'test_mape' in metrics.columns:
            row['MAPE (%)'] = metrics['test_mape'].mean()

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # 显示对比表格
    st.dataframe(comparison_df, use_container_width=True)

    # 可视化对比
    if len(comparison_df) > 0:
        st.markdown("#### 性能对比图")

        # 选择指标
        metric_cols = [c for c in comparison_df.columns if c != '模型']

        if len(metric_cols) > 0:
            selected_metric = st.selectbox("选择对比指标", metric_cols)

            # 柱状图
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(comparison_df['模型'], comparison_df[selected_metric], color='#1f77b4', alpha=0.7)

            # 添加数值标签
            for i, v in enumerate(comparison_df[selected_metric]):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

            apply_plot_style(ax,
                           title=f'{selected_metric} 对比',
                           xlabel='模型',
                           ylabel=selected_metric,
                           grid=True,
                           legend=False)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # 详细对比
    with st.expander("详细对比（按区域）"):
        # 选择指标
        available_metrics = ['test_mae', 'test_rmse', 'test_mape']
        selected_metric = st.selectbox(
            "选择指标",
            available_metrics,
            key="detailed_metric"
        )

        # 合并所有模型的数据
        merged_data = None

        for model_name, metrics in all_metrics.items():
            if selected_metric in metrics.columns and 'region' in metrics.columns:
                temp_df = metrics[['region', selected_metric]].copy()
                temp_df = temp_df.rename(columns={selected_metric: model_name})

                if merged_data is None:
                    merged_data = temp_df
                else:
                    merged_data = merged_data.merge(temp_df, on='region', how='outer')

        if merged_data is not None:
            st.dataframe(merged_data, use_container_width=True)

            # 选择区域绘制对比
            regions = merged_data['region'].tolist()
            selected_regions = st.multiselect(
                "选择区域查看对比",
                regions,
                default=regions[:5] if len(regions) >= 5 else regions,
                key="comp_regions"
            )

            if selected_regions:
                filtered_data = merged_data[merged_data['region'].isin(selected_regions)]

                # 绘制对比图
                fig, ax = plt.subplots(figsize=(12, 6))

                x = np.arange(len(selected_regions))
                width = 0.8 / len(all_metrics)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                for idx, model_name in enumerate(all_metrics.keys()):
                    if model_name in filtered_data.columns:
                        offset = width * idx - width * len(all_metrics) / 2
                        color = colors[idx % len(colors)]
                        ax.bar(x + offset, filtered_data[model_name],
                              width, label=model_name, color=color, alpha=0.8)

                ax.set_xticks(x)
                ax.set_xticklabels(selected_regions, rotation=45, ha='right')

                apply_plot_style(ax,
                               title=f'{selected_metric} 按区域对比',
                               xlabel='区域',
                               ylabel=selected_metric,
                               grid=True,
                               legend=True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


def show_detailed_analysis(prediction_files, metrics_files, output_path):
    """显示详细分析"""
    st.markdown("### 详细分析")

    # 选择模型
    model_names = [f.stem.replace("_predictions", "") for f in prediction_files]
    selected_model = st.selectbox("选择模型进行详细分析", model_names, key="detailed_model")

    # 加载数据
    pred_file = output_path / f"{selected_model}_predictions.csv"
    metrics_file = output_path / f"{selected_model}_test_metrics.csv"

    if not pred_file.exists():
        st.error(f"❌ 找不到预测文件：{pred_file}")
        return

    try:
        predictions = pd.read_csv(pred_file)

        # 预测统计分析
        st.markdown("#### 预测统计分析")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 预测值分布")

            fig, ax = plt.subplots(figsize=(10, 6))

            # 所有区域的预测值分布
            all_predictions = predictions.values.flatten()
            ax.hist(all_predictions, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')

            apply_plot_style(ax,
                           title='所有区域预测值分布',
                           xlabel='预测值',
                           ylabel='频数',
                           grid=True,
                           legend=False)

            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("##### 预测趋势")

            # 计算平均预测值随时间的变化
            avg_predictions = predictions.mean(axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(avg_predictions.values, linewidth=2, color='#1f77b4')

            apply_plot_style(ax,
                           title='平均预测值趋势',
                           xlabel='时间步',
                           ylabel='平均预测值',
                           grid=True,
                           legend=False)

            st.pyplot(fig)
            plt.close()

        # 区域分析
        if metrics_file.exists():
            st.markdown("---")
            st.markdown("#### 区域性能分析")

            metrics = pd.read_csv(metrics_file)

            if 'region' in metrics.columns and 'test_mae' in metrics.columns:
                # 最好和最差的区域
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 表现最好的区域（MAE）")

                    best_regions = metrics.nsmallest(5, 'test_mae')
                    st.dataframe(best_regions, use_container_width=True)

                with col2:
                    st.markdown("##### 表现最差的区域（MAE）")

                    worst_regions = metrics.nlargest(5, 'test_mae')
                    st.dataframe(worst_regions, use_container_width=True)

                # 误差分布
                st.markdown("##### 误差分布分析")

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

                if 'test_mae' in metrics.columns:
                    axes[0].hist(metrics['test_mae'], bins=30, edgecolor='black', alpha=0.7, color=colors[0])
                    apply_plot_style(axes[0],
                                   title='MAE 分布',
                                   xlabel='MAE',
                                   ylabel='频数',
                                   grid=True,
                                   legend=False)

                if 'test_rmse' in metrics.columns:
                    axes[1].hist(metrics['test_rmse'], bins=30, edgecolor='black', alpha=0.7, color=colors[1])
                    apply_plot_style(axes[1],
                                   title='RMSE 分布',
                                   xlabel='RMSE',
                                   ylabel='频数',
                                   grid=True,
                                   legend=False)

                if 'test_mape' in metrics.columns:
                    axes[2].hist(metrics['test_mape'], bins=30, edgecolor='black', alpha=0.7, color=colors[2])
                    apply_plot_style(axes[2],
                                   title='MAPE 分布',
                                   xlabel='MAPE (%)',
                                   ylabel='频数',
                                   grid=True,
                                   legend=False)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # 导出报告
        st.markdown("---")
        st.markdown("#### 导出分析报告")

        if st.button("生成报告"):
            generate_report(selected_model, predictions, metrics if metrics_file.exists() else None)

    except Exception as e:
        st.error(f"分析失败：{str(e)}")


def generate_report(model_name, predictions, metrics):
    """生成分析报告"""
    report_lines = []

    report_lines.append(f"# {model_name} 模型分析报告\n")
    report_lines.append(f"生成时间：{pd.Timestamp.now()}\n")
    report_lines.append("\n---\n")

    # 基本信息
    report_lines.append("## 基本信息\n")
    report_lines.append(f"- 预测步数：{len(predictions)}\n")
    report_lines.append(f"- 区域数量：{len(predictions.columns)}\n")
    report_lines.append(f"- 平均预测值：{predictions.mean().mean():.2f}\n")
    report_lines.append(f"- 预测值标准差：{predictions.std().mean():.2f}\n")
    report_lines.append("\n")

    # 评估指标
    if metrics is not None:
        report_lines.append("## 评估指标\n")

        if 'test_mae' in metrics.columns:
            report_lines.append(f"- 平均 MAE：{metrics['test_mae'].mean():.2f}\n")

        if 'test_rmse' in metrics.columns:
            report_lines.append(f"- 平均 RMSE：{metrics['test_rmse'].mean():.2f}\n")

        if 'test_mape' in metrics.columns:
            report_lines.append(f"- 平均 MAPE：{metrics['test_mape'].mean():.2f}%\n")

        report_lines.append("\n")

        # 最好和最差的区域
        if 'region' in metrics.columns and 'test_mae' in metrics.columns:
            report_lines.append("### 表现最好的区域（Top 5）\n")
            best = metrics.nsmallest(5, 'test_mae')
            for _, row in best.iterrows():
                report_lines.append(f"- {row['region']}: MAE = {row['test_mae']:.2f}\n")

            report_lines.append("\n### 表现最差的区域（Bottom 5）\n")
            worst = metrics.nlargest(5, 'test_mae')
            for _, row in worst.iterrows():
                report_lines.append(f"- {row['region']}: MAE = {row['test_mae']:.2f}\n")

    # 保存报告
    report_path = Path("output") / f"{model_name}_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    st.success(f"✅ 报告已保存到：{report_path}")

    # 提供下载
    with open(report_path, 'r', encoding='utf-8') as f:
        st.download_button(
            label="📥 下载报告",
            data=f.read(),
            file_name=f"{model_name}_analysis_report.md",
            mime='text/markdown'
        )
