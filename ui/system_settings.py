"""
系统设置页面
"""

import streamlit as st
import torch
from pathlib import Path
import sys
import shutil

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def show():
    """显示系统设置页面"""
    st.title("系统设置")

    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["系统信息", "文件管理", "缓存管理", "关于"])

    with tab1:
        show_system_info()

    with tab2:
        show_file_management()

    with tab3:
        show_cache_management()

    with tab4:
        show_about()


def show_system_info():
    """显示系统信息"""
    st.markdown("### 系统信息")

    # Python环境
    st.markdown("#### Python环境")

    col1, col2 = st.columns(2)

    with col1:
        import platform
        st.info(f"Python版本：{platform.python_version()}")
        st.info(f"操作系统：{platform.system()} {platform.release()}")

    with col2:
        st.info(f"架构：{platform.machine()}")
        st.info(f"处理器：{platform.processor()}")

    # PyTorch信息
    st.markdown("#### PyTorch信息")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"PyTorch版本：{torch.__version__}")

    with col2:
        cuda_available = torch.cuda.is_available()
        device_name = "CUDA" if cuda_available else "CPU"
        st.info(f"计算设备：{device_name}")

    with col3:
        if cuda_available:
            st.info(f"CUDA版本：{torch.version.cuda}")
        else:
            st.info("CUDA：不可用")

    # GPU信息
    if cuda_available:
        st.markdown("#### GPU信息")

        for i in range(torch.cuda.device_count()):
            with st.expander(f"GPU {i}: {torch.cuda.get_device_name(i)}"):
                props = torch.cuda.get_device_properties(i)

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**总显存：** {props.total_memory / 1024**3:.2f} GB")
                    st.write(f"**多处理器数量：** {props.multi_processor_count}")

                with col2:
                    st.write(f"**计算能力：** {props.major}.{props.minor}")

                    # 显示显存使用情况
                    if hasattr(torch.cuda, 'memory_allocated'):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        st.write(f"**已分配显存：** {allocated:.2f} GB")
                        st.write(f"**已保留显存：** {reserved:.2f} GB")

    # 已安装的关键包
    st.markdown("#### 已安装的关键包")

    packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'scikit-learn', 'lightgbm', 'streamlit'
    ]

    installed_packages = {}

    for package in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'Unknown')
            installed_packages[package] = version
        except ImportError:
            installed_packages[package] = "未安装"

    # 显示为表格
    import pandas as pd
    df_packages = pd.DataFrame(list(installed_packages.items()),
                               columns=['包名', '版本'])
    st.dataframe(df_packages, use_container_width=True, hide_index=True)


def show_file_management():
    """显示文件管理"""
    st.markdown("### 文件管理")

    # 数据集管理
    st.markdown("#### 数据集")

    dataset_path = Path("dataset")
    if dataset_path.exists():
        csv_files = list(dataset_path.glob("*.csv"))
        geojson_files = list(dataset_path.glob("*.geojson"))

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**CSV文件数量：** {len(csv_files)}")

            for f in csv_files:
                size_mb = f.stat().st_size / 1024 / 1024
                st.text(f"• {f.name} ({size_mb:.2f} MB)")

        with col2:
            st.write(f"**GeoJSON文件数量：** {len(geojson_files)}")

            for f in geojson_files:
                size_mb = f.stat().st_size / 1024 / 1024
                st.text(f"• {f.name} ({size_mb:.2f} MB)")
    else:
        st.warning("数据集目录不存在")

    # 输出文件管理
    st.markdown("---")
    st.markdown("#### 输出文件")

    output_path = Path("output")
    if output_path.exists():
        all_files = list(output_path.glob("*"))

        if all_files:
            st.write(f"**文件总数：** {len(all_files)}")

            # 按类型分组
            csv_files = [f for f in all_files if f.suffix == '.csv']
            png_files = [f for f in all_files if f.suffix == '.png']
            other_files = [f for f in all_files if f.suffix not in ['.csv', '.png']]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("CSV文件", len(csv_files))

            with col2:
                st.metric("PNG图片", len(png_files))

            with col3:
                st.metric("其他文件", len(other_files))

            # 文件列表
            with st.expander("查看所有文件"):
                import pandas as pd
                from datetime import datetime

                file_info = []
                for f in all_files:
                    size_mb = f.stat().st_size / 1024 / 1024
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)

                    file_info.append({
                        "文件名": f.name,
                        "类型": f.suffix,
                        "大小(MB)": f"{size_mb:.2f}",
                        "修改时间": mtime.strftime("%Y-%m-%d %H:%M:%S")
                    })

                df_files = pd.DataFrame(file_info)
                st.dataframe(df_files, use_container_width=True, hide_index=True)

            # 清理选项
            st.markdown("---")
            st.markdown("##### 清理操作")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("清理 CSV 文件", type="secondary"):
                    confirm = st.checkbox("确认删除所有CSV文件？")
                    if confirm and st.button("确定删除", key="del_csv"):
                        for f in csv_files:
                            f.unlink()
                        st.success(f"已删除 {len(csv_files)} 个 CSV 文件")
                        st.rerun()

            with col2:
                if st.button("清理 PNG 文件", type="secondary"):
                    confirm = st.checkbox("确认删除所有PNG文件？")
                    if confirm and st.button("确定删除", key="del_png"):
                        for f in png_files:
                            f.unlink()
                        st.success(f"已删除 {len(png_files)} 个 PNG 文件")
                        st.rerun()

            # 全部清空
            st.markdown("---")
            if st.button("清空输出目录", type="secondary"):
                confirm = st.checkbox("确认删除所有输出文件？此操作不可恢复！")
                if confirm and st.button("确定清空", key="del_all"):
                    for f in all_files:
                        try:
                            f.unlink()
                        except Exception as e:
                            st.error(f"删除 {f.name} 失败：{str(e)}")

                    st.success(f"已删除 {len(all_files)} 个文件")
                    st.rerun()
        else:
            st.info("输出目录为空")
    else:
        st.warning("输出目录不存在")


def show_cache_management():
    """显示缓存管理"""
    st.markdown("### 缓存管理")

    # Session State
    st.markdown("#### Session State")

    if st.session_state:
        st.write("**当前存储的数据：**")

        for key in st.session_state.keys():
            value = st.session_state[key]
            value_type = type(value).__name__

            # 显示简化信息
            if value_type == 'DataFrame':
                st.text(f"• {key}: DataFrame ({value.shape[0]} × {value.shape[1]})")
            elif value_type in ['str', 'int', 'float', 'bool']:
                st.text(f"• {key}: {value_type} = {value}")
            else:
                st.text(f"• {key}: {value_type}")

        # 清理按钮
        if st.button("清空 Session State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session State 已清空")
            st.rerun()
    else:
        st.info("Session State 为空")

    # Streamlit缓存
    st.markdown("---")
    st.markdown("#### Streamlit 缓存")

    st.info("Streamlit自动管理缓存，通常不需要手动清理")

    if st.button("清除Streamlit缓存"):
        st.cache_data.clear()
        st.success("缓存已清除")

    # PyTorch缓存
    if torch.cuda.is_available():
        st.markdown("---")
        st.markdown("#### GPU 显存缓存")

        col1, col2 = st.columns(2)

        with col1:
            allocated = torch.cuda.memory_allocated() / 1024**3
            st.metric("已分配显存", f"{allocated:.2f} GB")

        with col2:
            reserved = torch.cuda.memory_reserved() / 1024**3
            st.metric("已保留显存", f"{reserved:.2f} GB")

        if st.button("清理 GPU 缓存"):
            torch.cuda.empty_cache()
            st.success("GPU 缓存已清理")
            st.rerun()


def show_about():
    """显示关于页面"""
    st.markdown("### 关于")

    st.markdown("""
    ## 交通流量预测系统

    **版本：** 1.0.0

    **开发日期：** 2024年11月

    功能特性：

    - **数据预处理**
        - 数据加载与预览
        - 缺失值处理
        - 异常检测与处理
        - 数据探索与可视化

    - **模型训练**
        - 支持5种预测模型
        - 灵活的参数配置
        - 实时训练监控
        - 自动保存结果

    - **结果分析**
        - 预测结果可视化
        - 模型性能对比
        - 详细误差分析
        - 报告导出

    技术栈：

    - **前端框架：** Streamlit
    - **机器学习：** LightGBM, scikit-learn
    - **深度学习：** PyTorch
    - **数据处理：** Pandas, NumPy
    - **可视化：** Matplotlib, Seaborn

    支持的模型：

    1. **LightGBM** - 梯度提升树
    2. **LSTM** - 长短期记忆网络
    3. **N-HiTS** - 多尺度分层插值时序模型
    4. **TBATS** - 三角季节性指数平滑
    5. **Chronos** - 预训练时序大模型

    使用指南：

    1. **数据预处理**：加载数据并进行清洗
    2. **模型训练**：选择模型并配置参数
    3. **结果分析**：查看预测结果和性能指标

    相关链接：

    - [Streamlit文档](https://docs.streamlit.io)
    - [PyTorch文档](https://pytorch.org/docs)
    - [LightGBM文档](https://lightgbm.readthedocs.io)

    许可证：

    本项目仅供学习和研究使用。

    ---

    © 2024 交通流量预测系统
    """)

    # 系统健康检查
    st.markdown("---")
    st.markdown("### 系统健康检查")

    if st.button("运行健康检查"):
        with st.spinner("正在检查系统..."):
            checks = []

            # 检查数据集
            dataset_path = Path("dataset")
            checks.append({
                "项目": "数据集目录",
                "状态": "正常" if dataset_path.exists() else "缺失"
            })

            # 检查输出目录
            output_path = Path("output")
            output_path.mkdir(exist_ok=True)
            checks.append({
                "项目": "输出目录",
                "状态": "正常"
            })

            # 检查模型脚本
            predict_path = Path("predict")
            model_scripts = [
                "model_lightgbm.py",
                "model_lstm.py",
                "model_nhits.py",
                "model_tbats.py",
                "model_chronos.py"
            ]

            missing_scripts = []
            for script in model_scripts:
                if not (predict_path / script).exists():
                    missing_scripts.append(script)

            checks.append({
                "项目": "模型脚本",
                "状态": "全部存在" if not missing_scripts else f"缺少: {', '.join(missing_scripts)}"
            })

            # 检查关键包
            critical_packages = ['pandas', 'numpy', 'torch', 'streamlit']
            missing_packages = []

            for package in critical_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            checks.append({
                "项目": "关键依赖包",
                "状态": "全部安装" if not missing_packages else f"缺少: {', '.join(missing_packages)}"
            })

            # 显示检查结果
            import pandas as pd
            df_checks = pd.DataFrame(checks)
            st.dataframe(df_checks, use_container_width=True, hide_index=True)

            # 总体状态
            all_ok = all("缺少" not in check["状态"] for check in checks)
            if all_ok:
                st.success("系统运行正常")
            else:
                st.warning("系统存在一些问题，请查看上方详情")
