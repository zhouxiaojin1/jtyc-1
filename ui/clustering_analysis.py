"""
聚类分析页面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.clustering_analysis import TimeSeriesClustering
from utils.plot_config import setup_chinese_font

# 配置中文字体
setup_chinese_font()


def show():
    """显示聚类分析页面"""
    st.title("📊 聚类分析")

    st.markdown("""
    ### 功能说明
    对交通流量数据进行聚类分析，发现具有相似流量模式的区域。

    **支持的聚类方法：**
    - **K-Means**: 快速、适合大规模数据
    - **层次聚类**: 可生成树状图，层次关系清晰
    - **K-Shape**: 专门针对时间序列，基于形状相似度
    """)

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["数据准备", "聚类分析", "结果查看"])

    with tab1:
        show_data_preparation()

    with tab2:
        show_clustering_analysis()

    with tab3:
        show_results()


def show_data_preparation():
    """数据准备选项卡"""
    st.markdown("### 📁 数据准备")

    # 数据集选择
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        st.error("❌ 数据集文件夹不存在！")
        return

    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        st.warning("⚠️ 没有找到CSV数据文件")
        return

    # 选择数据集
    selected_file = st.selectbox(
        "选择数据集",
        csv_files,
        format_func=lambda x: x.name
    )

    if st.button("🔄 加载数据", type="primary"):
        try:
            with st.spinner("加载数据中..."):
                df = pd.read_csv(selected_file)

                # 保存到session state
                st.session_state['clustering_data'] = df
                st.session_state['clustering_data_path'] = str(selected_file)

                st.success(f"✅ 数据加载成功！形状: {df.shape}")

                # 显示数据预览
                st.markdown("#### 数据预览")
                st.dataframe(df.head(10), use_container_width=True, height=300)

                # 数据统计
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总行数", df.shape[0])
                with col2:
                    st.metric("总列数", df.shape[1])
                with col3:
                    time_col = df.columns[0]
                    region_cols = [col for col in df.columns if col != time_col]
                    st.metric("区域数", len(region_cols))
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("缺失率", f"{missing_pct:.2f}%")

        except Exception as e:
            st.error(f"❌ 加载数据失败: {str(e)}")

    # 数据选择
    if 'clustering_data' in st.session_state:
        st.markdown("---")
        st.markdown("#### 数据筛选")

        df = st.session_state['clustering_data']
        time_col = df.columns[0]
        all_regions = [col for col in df.columns if col != time_col]

        col1, col2 = st.columns(2)

        with col1:
            # 选择区域
            n_regions = st.slider(
                "选择区域数量",
                min_value=5,
                max_value=min(50, len(all_regions)),
                value=min(20, len(all_regions)),
                help="选择用于聚类的区域数量"
            )

            selected_regions = all_regions[:n_regions]

        with col2:
            # 选择时间范围
            max_steps = len(df)
            n_timesteps = st.slider(
                "选择时间步数",
                min_value=144,
                max_value=min(10080, max_steps),  # 最多一周
                value=min(1008, max_steps),
                step=144,
                help="选择用于聚类的时间步数（144=1天，1008=1周）"
            )

        # 筛选数据
        filtered_df = df[[time_col] + selected_regions].iloc[:n_timesteps]

        st.info(f"📊 筛选后的数据形状: {filtered_df.shape} (时间步={n_timesteps}, 区域数={n_regions})")

        # 保存筛选后的数据
        st.session_state['filtered_clustering_data'] = filtered_df
        st.session_state['time_col'] = time_col
        st.session_state['selected_regions'] = selected_regions


def show_clustering_analysis():
    """聚类分析选项卡"""
    st.markdown("### 🔍 聚类分析")

    if 'filtered_clustering_data' not in st.session_state:
        st.warning("⚠️ 请先在'数据准备'选项卡中加载数据")
        return

    df = st.session_state['filtered_clustering_data']
    time_col = st.session_state['time_col']

    # 聚类参数设置
    st.markdown("#### ⚙️ 参数设置")

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "聚类方法",
            ['kmeans', 'hierarchical', 'kshape'],
            format_func=lambda x: {
                'kmeans': 'K-Means',
                'hierarchical': '层次聚类',
                'kshape': 'K-Shape'
            }[x],
            help="选择聚类算法"
        )

    with col2:
        n_clusters = st.number_input(
            "聚类数 (k)",
            min_value=2,
            max_value=10,
            value=5,
            help="要分成几个聚类"
        )

    with col3:
        normalize = st.checkbox(
            "标准化数据",
            value=True,
            help="是否对数据进行标准化处理"
        )

    # 寻找最优k
    st.markdown("---")
    st.markdown("#### 🎯 寻找最优聚类数")

    col1, col2 = st.columns([1, 3])

    with col1:
        k_min = st.number_input("k 最小值", min_value=2, max_value=10, value=2)
        k_max = st.number_input("k 最大值", min_value=3, max_value=15, value=10)

        if st.button("🔍 寻找最优k", type="secondary"):
            try:
                with st.spinner("正在计算最优k值..."):
                    clustering = TimeSeriesClustering(
                        n_clusters=5,
                        method='kmeans',
                        normalize=normalize
                    )

                    output_dir = Path("output") / "clustering"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    fig, best_k = clustering.find_optimal_k(
                        df,
                        time_col=time_col,
                        k_range=range(k_min, k_max + 1),
                        save_path=str(output_dir / 'optimal_k.png')
                    )

                    st.session_state['optimal_k_fig'] = fig
                    st.session_state['best_k'] = best_k

                    st.success(f"✅ 推荐的最优聚类数: k = {best_k}")

            except Exception as e:
                st.error(f"❌ 计算失败: {str(e)}")

    with col2:
        if 'optimal_k_fig' in st.session_state:
            st.pyplot(st.session_state['optimal_k_fig'])

    # 执行聚类
    st.markdown("---")
    st.markdown("#### ▶️ 执行聚类")

    if st.button("🚀 开始聚类", type="primary"):
        try:
            with st.spinner(f"正在使用 {method} 方法进行聚类..."):
                # 创建聚类器
                clustering = TimeSeriesClustering(
                    n_clusters=n_clusters,
                    method=method,
                    normalize=normalize
                )

                # 执行聚类
                labels = clustering.fit(df, time_col=time_col)

                # 保存结果
                st.session_state['clustering_model'] = clustering
                st.session_state['clustering_labels'] = labels

                # 显示结果摘要
                st.success("✅ 聚类完成！")

                # 聚类摘要
                st.markdown("#### 📋 聚类摘要")
                summary_df = clustering.get_cluster_summary()
                st.dataframe(summary_df, use_container_width=True)

                # 评估指标
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "轮廓系数",
                        f"{clustering.metrics.get('silhouette', 0):.4f}",
                        help="范围[-1, 1]，越接近1越好"
                    )
                with col2:
                    st.metric(
                        "Davies-Bouldin指数",
                        f"{clustering.metrics.get('davies_bouldin', 0):.4f}",
                        help="越小越好"
                    )
                with col3:
                    st.metric(
                        "Calinski-Harabasz指数",
                        f"{clustering.metrics.get('calinski_harabasz', 0):.2f}",
                        help="越大越好"
                    )

                # 保存结果到文件
                output_dir = Path("output") / "clustering"
                output_dir.mkdir(parents=True, exist_ok=True)

                # 保存聚类结果
                region_names = st.session_state['selected_regions']
                cluster_result = pd.DataFrame({
                    'region': region_names,
                    'cluster': labels
                })
                cluster_result.to_csv(output_dir / 'cluster_result.csv', index=False)

                # 保存摘要
                summary_df.to_csv(output_dir / 'cluster_summary.csv', index=False)

                st.info(f"💾 聚类结果已保存到 {output_dir}")

        except Exception as e:
            st.error(f"❌ 聚类失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_results():
    """结果查看选项卡"""
    st.markdown("### 📈 聚类结果可视化")

    output_dir = Path("output") / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    st.markdown("---")
    st.markdown("#### 🖼️ 已生成图片")
    image_files = sorted(output_dir.glob("*.png"))
    if image_files:
        captions = {
            'optimal_k.png': '最优k分析',
            'cluster_centers.png': '聚类中心',
            'cluster_distribution.png': '聚类分布',
            'pca_visualization.png': 'PCA降维可视化',
            'dendrogram.png': '层次聚类树状图'
        }
        for img in image_files:
            cap = captions.get(img.name, img.name)
            st.image(str(img), caption=cap, use_column_width=True)
    else:
        st.info("暂无已生成图片")

    if 'clustering_model' not in st.session_state:
        st.warning("⚠️ 请先在'聚类分析'选项卡中执行聚类以使用交互式可视化")
        return

    clustering = st.session_state['clustering_model']
    df = st.session_state['filtered_clustering_data']
    time_col = st.session_state['time_col']

    viz_option = st.selectbox(
        "选择可视化类型",
        [
            "聚类中心",
            "聚类分布",
            "PCA降维可视化",
            "层次聚类树状图"
        ]
    )

    if viz_option == "聚类中心":
        st.markdown("#### 聚类中心曲线")
        st.markdown("显示每个聚类的中心（平均）模式")

        try:
            save_path = output_dir / 'cluster_centers.png'
            fig = clustering.plot_cluster_centers(save_path=str(save_path))
            st.pyplot(fig)
            st.success(f"💾 图表已保存到 {save_path}")
        except Exception as e:
            st.error(f"❌ 绘图失败: {str(e)}")

    elif viz_option == "聚类分布":
        st.markdown("#### 聚类分布")
        st.markdown("显示各聚类的样本数量分布")

        try:
            save_path = output_dir / 'cluster_distribution.png'
            fig = clustering.plot_cluster_distribution(save_path=str(save_path))
            st.pyplot(fig)
            st.success(f"💾 图表已保存到 {save_path}")
        except Exception as e:
            st.error(f"❌ 绘图失败: {str(e)}")

    elif viz_option == "PCA降维可视化":
        st.markdown("#### PCA降维可视化")
        st.markdown("使用主成分分析(PCA)将高维数据降到2维进行可视化")

        try:
            save_path = output_dir / 'pca_visualization.png'
            fig = clustering.plot_pca_visualization(df, time_col=time_col, save_path=str(save_path))
            st.pyplot(fig)
            st.success(f"💾 图表已保存到 {save_path}")
        except Exception as e:
            st.error(f"❌ 绘图失败: {str(e)}")

    elif viz_option == "层次聚类树状图":
        st.markdown("#### 层次聚类树状图")
        st.markdown("显示区域之间的层次关系")

        try:
            save_path = output_dir / 'dendrogram.png'
            fig = clustering.plot_dendrogram(df, time_col=time_col, save_path=str(save_path))
            st.pyplot(fig)
            st.success(f"💾 图表已保存到 {save_path}")
        except Exception as e:
            st.error(f"❌ 绘图失败: {str(e)}")

    # 聚类详情
    st.markdown("---")
    st.markdown("#### 🔍 聚类详情")

    if 'clustering_labels' in st.session_state:
        labels = st.session_state['clustering_labels']
        region_names = st.session_state['selected_regions']

        # 按聚类分组显示区域
        cluster_details = {}
        for i, region in enumerate(region_names):
            cluster_id = labels[i]
            if cluster_id not in cluster_details:
                cluster_details[cluster_id] = []
            cluster_details[cluster_id].append(region)

        # 显示每个聚类的区域
        for cluster_id in sorted(cluster_details.keys()):
            with st.expander(f"📌 聚类 {cluster_id} ({len(cluster_details[cluster_id])} 个区域)"):
                regions_text = ", ".join(cluster_details[cluster_id])
                st.write(regions_text)

    # 下载结果
    st.markdown("---")
    st.markdown("#### 💾 下载结果")

    col1, col2 = st.columns(2)

    with col1:
        # 下载聚类结果
        result_file = output_dir / 'cluster_result.csv'
        if result_file.exists():
            with open(result_file, 'rb') as f:
                st.download_button(
                    label="📥 下载聚类结果",
                    data=f,
                    file_name='cluster_result.csv',
                    mime='text/csv'
                )

    with col2:
        # 下载聚类摘要
        summary_file = output_dir / 'cluster_summary.csv'
        if summary_file.exists():
            with open(summary_file, 'rb') as f:
                st.download_button(
                    label="📥 下载聚类摘要",
                    data=f,
                    file_name='cluster_summary.csv',
                    mime='text/csv'
                )
