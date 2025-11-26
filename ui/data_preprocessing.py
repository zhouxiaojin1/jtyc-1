"""
数据预处理页面
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

from utils.plot_config import setup_chinese_font
setup_chinese_font()


def show():
    """显示数据预处理页面"""
    st.title("数据预处理")

    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["数据加载", "缺失值处理", "异常检测", "数据探索"])

    with tab1:
        show_data_loading()

    with tab2:
        show_missing_value_handling()

    with tab3:
        show_anomaly_detection()

    with tab4:
        show_data_exploration()


def show_data_loading():
    """数据加载选项卡"""
    st.markdown("### 数据加载")

    # 数据集选择
    dataset_path = Path("dataset")

    if not dataset_path.exists():
        st.error("数据集文件夹不存在！")
        return

    # 列出可用数据集
    csv_files = list(dataset_path.glob("*.csv"))

    if not csv_files:
        st.warning("未找到 CSV 数据文件")
        return

    # 选择数据集
    selected_file = st.selectbox(
        "选择数据集",
        [f.name for f in csv_files],
        index=0
    )

    # 加载数据
    if st.button("加载数据", type="primary"):
        with st.spinner("正在加载数据..."):
            try:
                df = pd.read_csv(dataset_path / selected_file)
                st.session_state['raw_data'] = df
                st.session_state['processed_data'] = df.copy()
                st.session_state['data_loaded'] = True
                st.session_state['selected_dataset_file'] = selected_file
                st.session_state['selected_dataset_path'] = str(dataset_path / selected_file)

                st.success(f"成功加载数据集：{selected_file}")

            except Exception as e:
                st.error(f"❌ 加载失败：{str(e)}")
                return

    # 显示数据概览
    if 'raw_data' in st.session_state:
        df = st.session_state['raw_data']

        st.markdown("---")
        st.markdown("### 数据概览")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总行数", f"{df.shape[0]:,}")

        with col2:
            st.metric("总列数", f"{df.shape[1]:,}")

        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("缺失值比例", f"{missing_pct:.2f}%")

        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("内存占用", f"{memory_mb:.2f} MB")

        # 数据预览
        st.markdown("### 🔍 数据预览")
        st.dataframe(df.head(100), use_container_width=True, height=300)

        # 列信息
        with st.expander("列信息详情"):
            col_info = pd.DataFrame({
                '列名': df.columns,
                '数据类型': df.dtypes.values,
                '非空值数': df.count().values,
                '缺失值数': df.isnull().sum().values,
                '缺失率(%)': (df.isnull().sum() / len(df) * 100).values
            })
            st.dataframe(col_info, use_container_width=True)

        # 基本统计信息
        with st.expander("基本统计信息"):
            st.dataframe(df.describe(), use_container_width=True)


def show_missing_value_handling():
    """缺失值处理选项卡"""
    st.markdown("### 缺失值处理")

    if 'raw_data' not in st.session_state:
        st.warning("请先加载数据集")
        return

    df = st.session_state['processed_data']

    # 缺失值统计
    st.markdown("#### 缺失值统计")

    missing_stats = pd.DataFrame({
        '列名': df.columns,
        '缺失值数': df.isnull().sum().values,
        '缺失率(%)': (df.isnull().sum() / len(df) * 100).values
    })
    missing_stats = missing_stats[missing_stats['缺失值数'] > 0].sort_values('缺失率(%)', ascending=False)

    if len(missing_stats) == 0:
        st.success("数据集中没有缺失值")
        return

    st.dataframe(missing_stats, use_container_width=True)

    # 缺失值可视化
    if len(missing_stats) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_stats_plot = missing_stats.head(20)
        ax.barh(missing_stats_plot['列名'], missing_stats_plot['缺失率(%)'])
        ax.set_xlabel('缺失率 (%)')
        ax.set_title('前20个缺失值最多的列')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # 处理方法选择
    st.markdown("---")
    st.markdown("#### 处理方法")

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "选择处理方法",
            ["前向填充 (Forward Fill)",
             "后向填充 (Backward Fill)",
             "线性插值 (Linear Interpolation)",
             "均值填充 (Mean)",
             "中位数填充 (Median)",
             "删除缺失行"]
        )

    with col2:
        # 选择要处理的列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if df.columns[0] in numeric_cols:
            numeric_cols.remove(df.columns[0])  # 移除时间列

        selected_cols = st.multiselect(
            "选择要处理的列（留空表示所有数值列）",
            numeric_cols
        )

        if not selected_cols:
            selected_cols = numeric_cols

    # 执行处理
    if st.button("执行处理", type="primary"):
        with st.spinner("正在处理缺失值..."):
            try:
                df_processed = df.copy()

                for col in selected_cols:
                    if col not in df_processed.columns:
                        continue

                    if method == "前向填充 (Forward Fill)":
                        df_processed[col] = df_processed[col].fillna(method='ffill')
                    elif method == "后向填充 (Backward Fill)":
                        df_processed[col] = df_processed[col].fillna(method='bfill')
                    elif method == "线性插值 (Linear Interpolation)":
                        df_processed[col] = df_processed[col].interpolate(method='linear')
                    elif method == "均值填充 (Mean)":
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    elif method == "中位数填充 (Median)":
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    elif method == "删除缺失行":
                        df_processed = df_processed.dropna(subset=[col])

                st.session_state['processed_data'] = df_processed

                # 显示处理结果
                missing_after = df_processed[selected_cols].isnull().sum().sum()
                st.success(f"处理完成！处理后缺失值数量：{missing_after}")

                # 对比前后
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("处理前缺失值", df[selected_cols].isnull().sum().sum())
                with col2:
                    st.metric("处理后缺失值", missing_after)

            except Exception as e:
                st.error(f"❌ 处理失败：{str(e)}")

    # 保存处理后的数据
    st.markdown("---")
    if st.button("保存处理后的数据"):
        output_path = Path("dataprecess")
        output_path.mkdir(exist_ok=True)

        output_file = output_path / "processed_data.csv"
        df.to_csv(output_file, index=False)

        st.success(f"数据已保存到：{output_file}")


def show_anomaly_detection():
    """异常检测选项卡"""
    st.markdown("### 异常检测")

    if 'processed_data' not in st.session_state:
        st.warning("请先加载数据集")
        return

    df = st.session_state['processed_data']

    # 选择要分析的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df.columns[0] in numeric_cols:
        numeric_cols.remove(df.columns[0])  # 移除时间列

    col1, col2 = st.columns(2)

    with col1:
        selected_col = st.selectbox("选择要检测的列", numeric_cols)

    with col2:
        method = st.selectbox(
            "检测方法",
            ["IQR方法 (四分位数)", "Z-Score方法 (标准差)", "MAD方法 (中位数绝对偏差)"]
        )

    # 参数设置
    if method == "IQR方法 (四分位数)":
        threshold = st.slider("IQR倍数", 1.0, 3.0, 1.5, 0.1)
    elif method == "Z-Score方法 (标准差)":
        threshold = st.slider("Z-Score阈值", 2.0, 4.0, 3.0, 0.1)
    else:  # MAD
        threshold = st.slider("MAD倍数", 2.0, 5.0, 3.0, 0.1)

    # 执行检测
    if st.button("开始检测", type="primary"):
        with st.spinner("正在检测异常值..."):
            try:
                series = df[selected_col].dropna()

                if method == "IQR方法 (四分位数)":
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    anomalies = (series < lower_bound) | (series > upper_bound)

                elif method == "Z-Score方法 (标准差)":
                    z_scores = np.abs((series - series.mean()) / series.std())
                    anomalies = z_scores > threshold

                else:  # MAD
                    median = series.median()
                    mad = np.median(np.abs(series - median))
                    modified_z_scores = 0.6745 * (series - median) / mad
                    anomalies = np.abs(modified_z_scores) > threshold

                # 统计结果
                n_anomalies = anomalies.sum()
                anomaly_rate = n_anomalies / len(series) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总数据点", len(series))
                with col2:
                    st.metric("异常值数量", n_anomalies)
                with col3:
                    st.metric("异常值比例", f"{anomaly_rate:.2f}%")

                # 可视化
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # 时间序列图
                ax1.plot(series.index, series.values, label='原始数据', alpha=0.7)
                ax1.scatter(series[anomalies].index, series[anomalies].values,
                           color='red', label='异常值', s=50, zorder=5)
                ax1.set_title(f'{selected_col} - 异常值检测')
                ax1.set_xlabel('索引')
                ax1.set_ylabel('值')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 箱线图
                ax2.boxplot(series.values, vert=False)
                ax2.scatter(series[anomalies].values,
                           np.ones(n_anomalies),
                           color='red', s=50, zorder=5, label='异常值')
                ax2.set_title('箱线图')
                ax2.set_xlabel('值')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 处理选项
                st.markdown("---")
                st.markdown("#### ⚙️ 异常值处理")

                handle_method = st.selectbox(
                    "选择处理方法",
                    ["不处理", "删除异常值", "用边界值替换", "用中位数替换", "用均值替换"]
                )

                if handle_method != "不处理" and st.button("应用处理"):
                    df_processed = df.copy()

                    if handle_method == "删除异常值":
                        df_processed = df_processed[~anomalies]
                    elif handle_method == "用边界值替换":
                        if method == "IQR方法 (四分位数)":
                            df_processed.loc[series < lower_bound, selected_col] = lower_bound
                            df_processed.loc[series > upper_bound, selected_col] = upper_bound
                    elif handle_method == "用中位数替换":
                        df_processed.loc[anomalies, selected_col] = series.median()
                    elif handle_method == "用均值替换":
                        df_processed.loc[anomalies, selected_col] = series.mean()

                    st.session_state['processed_data'] = df_processed
                    st.success(f"✅ 已应用处理方法：{handle_method}")

            except Exception as e:
                st.error(f"❌ 检测失败：{str(e)}")


def show_data_exploration():
    """数据探索选项卡"""
    st.markdown("### 📈 数据探索")

    if 'processed_data' not in st.session_state:
        st.warning("⚠️ 请先加载数据集")
        return

    df = st.session_state['processed_data']

    # 选择要探索的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df.columns[0] in numeric_cols:
        numeric_cols.remove(df.columns[0])  # 移除时间列

    # 探索选项
    exploration_type = st.selectbox(
        "选择探索类型",
        ["时间序列可视化", "分布分析", "相关性分析", "统计摘要"]
    )

    if exploration_type == "时间序列可视化":
        show_time_series_viz(df, numeric_cols)
    elif exploration_type == "分布分析":
        show_distribution_analysis(df, numeric_cols)
    elif exploration_type == "相关性分析":
        show_correlation_analysis(df, numeric_cols)
    elif exploration_type == "统计摘要":
        show_statistical_summary(df, numeric_cols)


def show_time_series_viz(df, numeric_cols):
    """时间序列可视化"""
    st.markdown("#### 📊 时间序列可视化")

    # 选择列
    selected_cols = st.multiselect(
        "选择要可视化的列（最多5列）",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )

    if len(selected_cols) > 5:
        st.warning("⚠️ 最多选择5列")
        selected_cols = selected_cols[:5]

    if not selected_cols:
        st.info("请选择至少一列")
        return

    # 采样选项
    sample_size = st.slider("显示数据点数", 100, min(10000, len(df)), min(1000, len(df)), 100)

    # 绘制
    if st.button("📊 生成图表"):
        fig, axes = plt.subplots(len(selected_cols), 1, figsize=(12, 4*len(selected_cols)))
        if len(selected_cols) == 1:
            axes = [axes]

        df_sample = df.iloc[-sample_size:]

        for idx, col in enumerate(selected_cols):
            axes[idx].plot(df_sample.index, df_sample[col].values, linewidth=1)
            axes[idx].set_title(f'{col}')
            axes[idx].set_xlabel('时间步')
            axes[idx].set_ylabel('值')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def show_distribution_analysis(df, numeric_cols):
    """分布分析"""
    st.markdown("#### 📊 分布分析")

    selected_col = st.selectbox("选择列", numeric_cols)

    if st.button("📊 分析分布"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 直方图
        axes[0].hist(df[selected_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'{selected_col} - 直方图')
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('频数')
        axes[0].grid(True, alpha=0.3)

        # 箱线图
        axes[1].boxplot(df[selected_col].dropna(), vert=True)
        axes[1].set_title(f'{selected_col} - 箱线图')
        axes[1].set_ylabel('值')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 统计信息
        st.markdown("##### 📊 统计信息")
        stats = df[selected_col].describe()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("均值", f"{stats['mean']:.2f}")
            st.metric("最小值", f"{stats['min']:.2f}")

        with col2:
            st.metric("中位数", f"{stats['50%']:.2f}")
            st.metric("25%分位", f"{stats['25%']:.2f}")

        with col3:
            st.metric("标准差", f"{stats['std']:.2f}")
            st.metric("75%分位", f"{stats['75%']:.2f}")

        with col4:
            st.metric("最大值", f"{stats['max']:.2f}")
            skew = df[selected_col].skew()
            st.metric("偏度", f"{skew:.2f}")


def show_correlation_analysis(df, numeric_cols):
    """相关性分析"""
    st.markdown("#### 📊 相关性分析")

    # 选择要分析的列
    selected_cols = st.multiselect(
        "选择要分析的列（留空表示所有列）",
        numeric_cols,
        default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
    )

    if not selected_cols:
        selected_cols = numeric_cols

    if len(selected_cols) < 2:
        st.warning("⚠️ 请至少选择2列进行相关性分析")
        return

    if len(selected_cols) > 20:
        st.warning("⚠️ 选择的列过多，将只显示前20列")
        selected_cols = selected_cols[:20]

    if st.button("📊 计算相关性"):
        with st.spinner("正在计算相关性..."):
            # 计算相关系数矩阵
            corr_matrix = df[selected_cols].corr()

            # 绘制热力图
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('相关性热力图')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 显示高相关性的列对
            st.markdown("##### 🔍 高相关性列对（|相关系数| > 0.7）")

            # 提取上三角矩阵（避免重复）
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            '列1': corr_matrix.columns[i],
                            '列2': corr_matrix.columns[j],
                            '相关系数': corr_val
                        })

            if high_corr:
                high_corr_df = pd.DataFrame(high_corr).sort_values('相关系数',
                                                                    key=abs,
                                                                    ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
            else:
                st.info("没有发现高相关性的列对")


def show_statistical_summary(df, numeric_cols):
    """统计摘要"""
    st.markdown("#### 📊 统计摘要")

    # 完整统计信息
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    # 额外统计量
    st.markdown("##### 📈 额外统计量")

    extra_stats = pd.DataFrame({
        '列名': numeric_cols,
        '偏度': [df[col].skew() for col in numeric_cols],
        '峰度': [df[col].kurtosis() for col in numeric_cols],
        '变异系数': [df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                      for col in numeric_cols]
    })

    st.dataframe(extra_stats, use_container_width=True)
