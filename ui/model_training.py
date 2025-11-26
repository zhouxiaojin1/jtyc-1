"""
模型训练页面
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import subprocess
import threading
import queue
import time
import json
from typing import Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def show():
    """显示模型训练页面"""
    st.title("模型训练")

    # 模型选择
    st.markdown("### 选择模型")

    model_descriptions = {
        "LightGBM": {
            "description": "梯度提升树模型，训练速度快，适合快速验证",
            "pros": "速度快、鲁棒、易调参",
            "cons": "多步递归可能误差累积",
            "script": "model_lightgbm.py"
        },
        "N-HiTS": {
            "description": "深度学习模型，长期预测效果好",
            "pros": "多季节建模强、推断快",
            "cons": "需要GPU加速",
            "script": "model_nhits.py"
        },
        "Prophet": {
            "description": "Facebook时间序列模型，快速且准确（推荐）",
            "pros": "速度极快、自动处理多季节性、可解释性强、高精度",
            "cons": "单变量模型，不利用区域间关联",
            "script": "model_prophet.py"
        },
        "Chronos": {
            "description": "预训练时间序列模型，零样本预测",
            "pros": "即用即准、泛化能力强",
            "cons": "模型较大，资源占用高",
            "script": "model_chronos.py"
        }
    }

    config_path = Path("config") / "training_config.json"
    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    if 'train_params_by_model' not in st.session_state:
        st.session_state['train_params_by_model'] = {}
    if not st.session_state.get('selected_model'):
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                model_name_last = cfg.get('model_name')
                if model_name_last and model_name_last in model_descriptions:
                    st.session_state['selected_model'] = model_name_last
                    st.session_state['model_script'] = model_descriptions[model_name_last]['script']
                history = cfg.get('history', {})
                mp = history.get('model_params_by_model', {})
                tp = history.get('train_params_by_model', {})
                if isinstance(mp, dict):
                    st.session_state['model_params_by_model'].update(mp)
                if isinstance(tp, dict):
                    st.session_state['train_params_by_model'].update(tp)
                if model_name_last:
                    mpl = cfg.get('model_params', {})
                    tpl = cfg.get('train_params', {})
                    if isinstance(mpl, dict):
                        st.session_state['model_params_by_model'][model_name_last] = mpl
                    if isinstance(tpl, dict):
                        st.session_state['train_params_by_model'][model_name_last] = tpl
            except Exception:
                pass

    # 显示模型卡片
    cols = st.columns(3)
    selected_model = None

    for idx, (model_name, info) in enumerate(model_descriptions.items()):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                <div style="padding: 1rem; border: 1px solid #ccc; border-radius: 0; height: 240px;">
                    <h3 style="margin:0;">{model_name}</h3>
                    <p><strong>简介：</strong>{info['description']}</p>
                    <p><strong>优点：</strong>{info['pros']}</p>
                    <p><strong>缺点：</strong>{info['cons']}</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"选择 {model_name}", key=f"select_{model_name}", use_container_width=True):
                    selected_model = model_name
                    st.session_state['selected_model'] = model_name
                    st.session_state['model_script'] = info['script']

    # 显示选中的模型
    if 'selected_model' in st.session_state:
        st.markdown("---")
        st.success(f"已选择模型：{st.session_state['selected_model']}")

        # 参数配置
        show_model_parameters()

        # 训练控制
        show_training_control()


def show_model_parameters():
    """显示模型参数配置"""
    st.markdown("### 模型参数配置")

    model_name = st.session_state['selected_model']
    last_train = st.session_state.get('train_params_by_model', {}).get(model_name, {})

    # 通用参数
    col1, col2 = st.columns(2)

    with col1:
        prediction_length = st.number_input(
            "预测步数",
            min_value=1,
            max_value=2016,
            value=clamp_int(last_train.get('prediction_length', 1008), 1, 2016, 1008),
            step=1,
            help="要预测的未来时间步数"
        )

    with col2:
        context_length = st.number_input(
            "历史窗口长度",
            min_value=100,
            max_value=5000,
            value=clamp_int(last_train.get('context_length', 2016), 100, 5000, 2016),
            step=100,
            help="用于预测的历史数据长度"
        )

    # 数据划分
    train_ratio = st.slider(
        "训练集比例",
        min_value=0.5,
        max_value=0.95,
        value=clamp_float(last_train.get('train_ratio', 0.9), 0.5, 0.95, 0.9),
        step=0.05,
        help="训练集占总数据的比例"
    )

    # 模型特定参数
    st.markdown("#### 模型特定参数")

    if model_name == "LightGBM":
        show_lightgbm_params()
    elif model_name == "N-HiTS":
        show_nhits_params()
    elif model_name == "Prophet":
        show_prophet_params()
    elif model_name == "Chronos":
        show_chronos_params()

    # 保存参数到session state
    if 'train_params_by_model' not in st.session_state:
        st.session_state['train_params_by_model'] = {}
    st.session_state['train_params_by_model'][model_name] = {
        'prediction_length': prediction_length,
        'context_length': context_length,
        'train_ratio': train_ratio
    }


def show_lightgbm_params():
    """LightGBM参数"""
    col1, col2, col3 = st.columns(3)
    saved = st.session_state.get('model_params_by_model', {}).get('LightGBM', {})

    with col1:
        n_estimators = st.number_input("树的数量", 100, 2000, clamp_int(saved.get('n_estimators', 500), 100, 2000, 500), 100)
        learning_rate = st.number_input("学习率", 0.01, 0.3, clamp_float(saved.get('learning_rate', 0.05), 0.01, 0.3, 0.05), 0.01)

    with col2:
        max_depth = st.number_input("最大深度", 3, 15, clamp_int(saved.get('max_depth', 7), 3, 15, 7), 1)
        num_leaves = st.number_input("叶子节点数", 10, 100, clamp_int(saved.get('num_leaves', 31), 10, 100, 31), 5)

    with col3:
        min_child_samples = st.number_input("最小子节点样本数", 5, 50, clamp_int(saved.get('min_child_samples', 20), 5, 50, 20), 5)
        subsample = st.slider("采样比例", 0.5, 1.0, clamp_float(saved.get('subsample', 0.8), 0.5, 1.0, 0.8), 0.1)

    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    st.session_state['model_params_by_model']['LightGBM'] = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'subsample': subsample
    }

def show_random_forest_params():
    col1, col2, col3 = st.columns(3)
    saved = st.session_state.get('model_params_by_model', {}).get('RandomForest', {})
    with col1:
        n_estimators = st.number_input("树的数量", 100, 2000, clamp_int(saved.get('n_estimators', 300), 100, 2000, 300), 50)
        max_depth = st.number_input("最大深度(-1为None)", -1, 50, clamp_int(saved.get('max_depth', -1), -1, 50, -1), 1)
    with col2:
        min_samples_split = st.number_input("最小分裂样本数", 2, 20, clamp_int(saved.get('min_samples_split', 2), 2, 20, 2), 1)
        min_samples_leaf = st.number_input("最小叶子样本数", 1, 20, clamp_int(saved.get('min_samples_leaf', 1), 1, 20, 1), 1)
    with col3:
        max_features = st.selectbox("特征采样", ["sqrt", "log2"], index=["sqrt","log2"].index(str(saved.get('max_features','sqrt'))))
        n_jobs = st.number_input("并行进程数", -1, 8, clamp_int(saved.get('n_jobs', -1), -1, 8, -1), 1)
    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    st.session_state['model_params_by_model']['RandomForest'] = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'n_jobs': n_jobs
    }


def show_lstm_params():
    """LSTM参数"""
    col1, col2, col3 = st.columns(3)
    saved = st.session_state.get('model_params_by_model', {}).get('LSTM', {})

    with col1:
        hidden_size = st.number_input("隐藏层大小", 32, 512, clamp_int(saved.get('hidden_size', 128), 32, 512, 128), 32)
        num_layers = st.number_input("LSTM层数", 1, 4, clamp_int(saved.get('num_layers', 2), 1, 4, 2), 1)

    with col2:
        dropout = st.slider("Dropout率", 0.0, 0.5, clamp_float(saved.get('dropout', 0.2), 0.0, 0.5, 0.2), 0.05)
        batch_size = st.number_input("批次大小", 16, 256, clamp_int(saved.get('batch_size', 64), 16, 256, 64), 16)

    with col3:
        epochs = st.number_input("训练轮数", 10, 200, clamp_int(saved.get('epochs', 50), 10, 200, 50), 10)
        learning_rate = st.number_input("学习率", 0.0001, 0.01, clamp_float(saved.get('learning_rate', 0.001), 0.0001, 0.01, 0.001), 0.0001, format="%.4f")

    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    st.session_state['model_params_by_model']['LSTM'] = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate
    }


def show_nhits_params():
    """N-HiTS参数"""
    col1, col2, col3 = st.columns(3)
    saved = st.session_state.get('model_params_by_model', {}).get('N-HiTS', {})

    with col1:
        n_blocks = st.number_input("块数量", 1, 5, clamp_int(saved.get('n_blocks', 3), 1, 5, 3), 1)
        n_layers = st.number_input("每块层数", 1, 4, clamp_int(saved.get('n_layers', 2), 1, 4, 2), 1)

    with col2:
        hidden_size = st.number_input("隐藏层大小", 128, 1024, clamp_int(saved.get('hidden_size', 512), 128, 1024, 512), 128)
        batch_size = st.number_input("批次大小", 16, 256, clamp_int(saved.get('batch_size', 64), 16, 256, 64), 16)

    with col3:
        epochs = st.number_input("训练轮数", 10, 200, clamp_int(saved.get('epochs', 50), 10, 200, 50), 10)
        learning_rate = st.number_input("学习率", 0.0001, 0.01, clamp_float(saved.get('learning_rate', 0.001), 0.0001, 0.01, 0.001), 0.0001, format="%.4f")

    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    st.session_state['model_params_by_model']['N-HiTS'] = {
        'n_blocks': n_blocks,
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate
    }


def show_prophet_params():
    """Prophet参数"""
    col1, col2, col3 = st.columns(3)
    saved = st.session_state.get('model_params_by_model', {}).get('Prophet', {})

    with col1:
        seasonality_mode_options = ['multiplicative', 'additive']
        default_mode = str(saved.get('seasonality_mode', 'multiplicative'))
        seasonality_mode = st.selectbox(
            "季节性模式",
            seasonality_mode_options,
            index=seasonality_mode_options.index(default_mode) if default_mode in seasonality_mode_options else 0,
            help="multiplicative适合交通流量，additive适合温度等"
        )
        daily_seasonality = st.checkbox("日周期", value=bool(saved.get('daily_seasonality', True)))

    with col2:
        weekly_seasonality = st.checkbox("周周期", value=bool(saved.get('weekly_seasonality', True)))
        yearly_seasonality = st.checkbox("年周期", value=bool(saved.get('yearly_seasonality', False)),
                                        help="需要至少1年数据")

    with col3:
        changepoint_prior_scale = st.slider(
            "趋势灵活度",
            0.001, 0.5,
            clamp_float(saved.get('changepoint_prior_scale', 0.05), 0.001, 0.5, 0.05),
            0.001,
            help="控制趋势变化点的灵活度，越大越灵活"
        )
        seasonality_prior_scale = st.slider(
            "季节性灵活度",
            0.01, 10.0,
            clamp_float(saved.get('seasonality_prior_scale', 10.0), 0.01, 10.0, 10.0),
            0.1,
            help="控制季节性的灵活度，越大越灵活"
        )

    n_jobs = st.number_input("并行训练进程数", 1, 8, clamp_int(saved.get('n_jobs', 2), 1, 8, 2), 1,
                             help="多个区域并行训练")

    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    st.session_state['model_params_by_model']['Prophet'] = {
        'seasonality_mode': seasonality_mode,
        'daily_seasonality': daily_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'yearly_seasonality': yearly_seasonality,
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'n_jobs': n_jobs
    }


def show_chronos_params():
    """Chronos参数"""
    col1, col2, col3 = st.columns(3)
    saved = st.session_state.get('model_params_by_model', {}).get('Chronos', {})

    with col1:
        options = ["tiny", "mini", "small", "base", "large"]
        default_size = str(saved.get('model_size', 'small'))
        model_size = st.selectbox(
            "模型大小",
            options,
            index=options.index(default_size) if default_size in options else 2,
            help="模型越大效果越好，但资源消耗也越大"
        )

    with col2:
        num_samples = st.number_input("采样数量", 10, 100, clamp_int(saved.get('num_samples', 20), 10, 100, 20), 10, help="用于概率预测的采样数")

    with col3:
        temperature = st.slider("采样温度", 0.5, 2.0, clamp_float(saved.get('temperature', 1.0), 0.5, 2.0, 1.0), 0.1, help="控制预测的随机性")

    use_median = st.checkbox("使用中位数（否则使用均值）", value=bool(saved.get('use_median', True)))

    if 'model_params_by_model' not in st.session_state:
        st.session_state['model_params_by_model'] = {}
    st.session_state['model_params_by_model']['Chronos'] = {
        'model_size': model_size,
        'num_samples': num_samples,
        'temperature': temperature,
        'use_median': use_median
    }


def show_training_control():
    """显示训练控制面板"""
    st.markdown("---")
    st.markdown("### 训练控制")

    # 检查数据
    data_status = "✅ 数据已加载" if 'processed_data' in st.session_state else "❌ 未加载数据"
    model_status = f"✅ 已选择：{st.session_state['selected_model']}" if 'selected_model' in st.session_state else "❌ 未选择模型"

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"数据状态：{data_status}")
    with col2:
        st.info(f"模型状态：{model_status}")

    # 训练按钮
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("开始训练", type="primary", use_container_width=True,
                     disabled='selected_model' not in st.session_state):
            start_training()

    with col2:
        if st.button("暂停", use_container_width=True, disabled=True):
            st.warning("暂停功能开发中")

    with col3:
        if st.button("停止", use_container_width=True, disabled=True):
            st.warning("停止功能开发中")

    # 训练进度
    if 'training_status' in st.session_state and st.session_state['training_status'] == 'running':
        show_training_progress()


def start_training():
    """开始训练模型"""
    st.session_state['training_status'] = 'running'
    st.session_state['training_logs'] = []

    # 获取模型脚本路径
    script_path = Path("predict") / st.session_state['model_script']

    if not script_path.exists():
        st.error(f"❌ 模型脚本不存在：{script_path}")
        st.session_state['training_status'] = 'failed'
        return

    # 保存训练参数到配置文件
    import json
    config_path = Path("config") / "training_config.json"
    config_path.parent.mkdir(exist_ok=True)

    model_name = st.session_state['selected_model']
    mp_all = st.session_state.get('model_params_by_model', {})
    tp_all = st.session_state.get('train_params_by_model', {})
    mp_cur = mp_all.get(model_name, {})
    tp_cur = tp_all.get(model_name, {})

    cfg_existing = {}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_existing = json.load(f)
        except Exception:
            cfg_existing = {}

    history = cfg_existing.get('history', {})
    h_mp = history.get('model_params_by_model', {})
    h_tp = history.get('train_params_by_model', {})
    h_dp = history.get('dataset_path_by_model', {})
    if not isinstance(h_mp, dict):
        h_mp = {}
    if not isinstance(h_tp, dict):
        h_tp = {}
    if not isinstance(h_dp, dict):
        h_dp = {}

    # 组合配置，加入数据集路径（来自数据预处理页选择）
    dataset_path = st.session_state.get('selected_dataset_path')
    if not dataset_path:
        # 回退到默认数据集
        dataset_path = str(Path('dataset') / 'milano_traffic_nid.csv')

    # 更新历史记录
    h_mp[model_name] = mp_cur
    h_tp[model_name] = tp_cur
    h_dp[model_name] = dataset_path

    config = {
        'model_name': model_name,
        'model_params': mp_cur,
        'train_params': tp_cur,
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_path': dataset_path,
        'history': {
            'model_params_by_model': h_mp,
            'train_params_by_model': h_tp,
            'dataset_path_by_model': h_dp
        }
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    st.info(f"训练参数已保存到: {config_path}")

    # 显示训练信息
    with st.spinner(f"正在启动 {st.session_state['selected_model']} 模型训练..."):
        st.info(f"执行脚本：{script_path}")

        # 创建训练进度容器
        progress_container = st.container()

        with progress_container:
            st.markdown("#### 训练进度")

            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 日志输出
            log_container = st.expander("训练日志", expanded=True)

            # 执行训练脚本
            try:
                with log_container:
                    log_area = st.empty()

                    # 使用conda环境中的Python运行脚本
                    # 获取当前conda环境的Python路径
                    import sys
                    import os
                    python_path = sys.executable

                    # 确保子进程继承当前的环境变量
                    env = os.environ.copy()

                    # 传递配置文件路径作为命令行参数
                    process = subprocess.Popen(
                        [python_path, str(script_path), "--config", str(config_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        env=env
                    )

                    logs = []
                    for line in process.stdout:
                        logs.append(line.strip())
                        log_area.text_area(
                            "输出",
                            value='\n'.join(logs[-50:]),  # 只显示最近50行
                            height=300,
                            key=f"log_{len(logs)}"
                        )

                        # 更新进度（这里简单模拟，实际需要从脚本输出解析）
                        if "epoch" in line.lower() or "预测" in line:
                            progress = min(len(logs) / 100, 0.99)
                            progress_bar.progress(progress)
                            status_text.text(f"进度：{int(progress*100)}% - {line.strip()}")

                    # 等待进程结束
                    process.wait()

                    if process.returncode == 0:
                        progress_bar.progress(1.0)
                        status_text.text("进度：100% - 训练完成！")
                        st.success("模型训练成功完成")
                        st.session_state['training_status'] = 'completed'

                        # 显示输出文件
                        show_output_files()
                    else:
                        reason = explain_exit_code(process.returncode)
                        st.error(f"训练失败，退出代码：{process.returncode} - {reason}")
                        st.markdown("#### 错误摘要")
                        tail = '\n'.join(logs[-100:]) if logs else "无日志输出"
                        st.code(tail)
                        from datetime import datetime
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = f"{st.session_state['selected_model']}_training_{ts}.log"
                        st.download_button(
                            label="下载完整日志",
                            data='\n'.join(logs) if logs else "",
                            file_name=fname,
                            mime="text/plain"
                        )
                        st.session_state['training_status'] = 'failed'

            except Exception as e:
                st.error(f"❌ 训练过程出错：{str(e)}")
                st.session_state['training_status'] = 'failed'


def show_training_progress():
    """显示训练进度"""
    st.markdown("#### 训练中...")

    # 简单的进度显示
    progress_bar = st.progress(0)

    # 这里可以添加实时进度更新逻辑
    # 实际应用中需要从训练脚本获取进度信息


def show_output_files():
    """显示输出文件"""
    st.markdown("#### 生成的文件")

    output_path = Path("output")
    if output_path.exists():
        files = sorted(output_path.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

        if files:
            # 只显示最近生成的文件
            recent_files = files[:10]

            file_info = []
            for f in recent_files:
                size_mb = f.stat().st_size / 1024 / 1024
                file_info.append({
                    "文件名": f.name,
                    "大小": f"{size_mb:.2f} MB",
                    "类型": f.suffix,
                    "修改时间": time.ctime(f.stat().st_mtime)
                })

            df_files = pd.DataFrame(file_info)
            st.dataframe(df_files, use_container_width=True)

            # 下载按钮
            st.markdown("##### 下载文件")
            for f in recent_files:
                if f.suffix in ['.csv', '.txt', '.json']:
                    try:
                        with open(f, 'rb') as file:
                            st.download_button(
                                label=f"下载 {f.name}",
                                data=file,
                                file_name=f.name,
                                mime='application/octet-stream',
                                key=f"download_{f.name}"
                            )
                    except Exception as e:
                        st.error(f"无法读取文件 {f.name}: {str(e)}")
        else:
            st.info("暂无输出文件")
    else:
        st.info("输出目录不存在")
def clamp_int(val: Any, min_v: int, max_v: int, default: int) -> int:
    try:
        v = int(val)
    except Exception:
        return default
    if v < min_v:
        return min_v
    if v > max_v:
        return max_v
    return v

def clamp_float(val: Any, min_v: float, max_v: float, default: float) -> float:
    try:
        v = float(val)
    except Exception:
        return default
    if v < min_v:
        return min_v
    if v > max_v:
        return max_v
    return v
def explain_exit_code(code: int) -> str:
    mapping = {
        0: "成功",
        1: "一般性错误",
        2: "命令使用错误",
        3221225477: "访问冲突 0xC0000005（常见于内存/显存问题或非法内存访问）",
        3221225786: "整数除零 0xC0000094",
        3221226505: "栈溢出 0xC00000FD"
    }
    return mapping.get(code, "未知错误")
