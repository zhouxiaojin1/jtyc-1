"""
交通流量预测系统 - 可视化界面
主入口文件
"""

import streamlit as st
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 页面配置
st.set_page_config(
    page_title="交通流量预测系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Serif SC','Songti SC','SimSun',serif;
        color: #333;
        background: #f9f6f2;
    }
    [data-testid="stSidebar"] {
        background: #f9f6f2;
        color: #333;
    }
    .main-header {
        font-size: 28px;
        font-weight: 600;
        color: #333;
        text-align: center;
        margin: 1.5rem 0;
    }
    .sub-header {
        font-size: 18px;
        font-weight: 600;
        color: #333;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: #555;
        color: #fff;
        border-radius: 0;
        border: 1px solid #555;
    }
    .stButton>button:hover {
        background: #333;
        border-color: #333;
    }
    .stAlert {
        background-color: #f1eee7;
        color: #333;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# 主页面
def main():
    # 标题
    st.markdown('<div class="main-header">交通流量预测系统</div>', unsafe_allow_html=True)

    # 侧边栏导航
    st.sidebar.title("导航菜单")
    page = st.sidebar.radio(
        "选择功能模块",
        ["首页", "数据预处理", "聚类分析", "模型训练", "结果分析", "系统设置"]
    )

    # 根据选择显示不同页面
    if page == "首页":
        show_home()
    elif page == "数据预处理":
        from ui import data_preprocessing
        data_preprocessing.show()
    elif page == "聚类分析":
        from ui import clustering_analysis
        clustering_analysis.show()
    elif page == "模型训练":
        from ui import model_training
        model_training.show()
    elif page == "结果分析":
        from ui import result_analysis
        result_analysis.show()
    elif page == "系统设置":
        from ui import system_settings
        system_settings.show()


def show_home():
    """显示首页"""

    # 欢迎信息
    st.markdown("""
    ### 欢迎使用交通流量预测系统

    本系统提供完整的时间序列预测流程，从数据预处理到模型训练再到结果分析。

    **主要功能：**
    - 数据预处理：缺失值处理、异常检测、数据可视化
    - 聚类分析：时间序列聚类、模式发现、区域分组
    - 模型训练：支持多种预测模型（LightGBM、N-HiTS、Chronos、Prophet）
    - 结果分析：模型性能对比、预测结果可视化
    - 系统设置：参数配置、模型管理
    """)

    # 快速开始指南
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="sub-header">步骤 1</div>', unsafe_allow_html=True)
        st.info("数据预处理\n\n加载数据集，处理缺失值和异常值，探索数据特征")
        if st.button("前往数据预处理"):
            st.session_state['page'] = "数据预处理"
            st.rerun()

    with col2:
        st.markdown('<div class="sub-header">步骤 2</div>', unsafe_allow_html=True)
        st.info("模型训练\n\n选择预测模型，配置参数，训练并保存模型")
        if st.button("前往模型训练"):
            st.session_state['page'] = "模型训练"
            st.rerun()

    with col3:
        st.markdown('<div class="sub-header">步骤 3</div>', unsafe_allow_html=True)
        st.info("结果分析\n\n查看预测结果，对比模型性能，生成报告")
        if st.button("前往结果分析"):
            st.session_state['page'] = "结果分析"
            st.rerun()

    # 系统状态
    st.markdown('<div class="sub-header">系统状态</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ds_name = "未知"
        try:
            cfg_path = Path("config") / "training_config.json"
            if cfg_path.exists():
                import json
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                ds = cfg.get('dataset_path')
                if ds:
                    ds_name = Path(ds).name
        except Exception:
            pass
        st.metric("数据集", ds_name)

    with col2:
        st.metric("可用模型", "5")

    with col3:
        # 检查是否有训练过的模型
        output_path = Path("output")
        trained_models = 0
        if output_path.exists():
            trained_models = len(list(output_path.glob("*_predictions.csv")))
        st.metric("已训练模型", str(trained_models))

    with col4:
        try:
            import torch
            device = "GPU" if torch.cuda.is_available() else "CPU"
        except ImportError:
            device = "CPU"
        st.metric("计算设备", device)

    # 最近活动
    st.markdown('<div class="sub-header">最近活动</div>', unsafe_allow_html=True)

    try:
        if output_path.exists():
            import os
            from datetime import datetime

            # 获取最近修改的文件
            files = []
            for f in output_path.glob("*"):
                if f.is_file():
                    try:
                        files.append((f.name, datetime.fromtimestamp(os.path.getmtime(f))))
                    except:
                        continue

            if files:
                files.sort(key=lambda x: x[1], reverse=True)

                st.markdown("最近生成的文件：")
                for filename, timestamp in files[:5]:
                    st.text(f"• {filename} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("暂无活动记录")
        else:
            st.info("暂无活动记录")
    except Exception as e:
        st.info("暂无活动记录")

    # 帮助信息
    with st.expander("使用提示"):
        st.markdown("""
        如何开始：

        1. 进入"数据预处理"页面，加载并处理数据
        2. 在"模型训练"页面选择模型并配置参数
        3. 在"结果分析"页面查看预测结果和性能指标

        模型说明：

        - LightGBM：梯度提升树，训练速度快
        - N-HiTS：深度学习模型，长期预测效果好
        - Chronos：预训练模型，零样本预测
        - Prophet：统计模型，自动处理多季节性

        性能说明：

        - CPU模式适合小规模数据
        - GPU模式可加速训练
        """)


if __name__ == "__main__":
    main()
