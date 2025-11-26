"""
GPU环境检查工具

运行此脚本检查各模型的GPU支持情况
"""

import sys

print("=" * 60)
print("GPU环境检查")
print("=" * 60)

# 1. 检查PyTorch和CUDA
print("\n1. PyTorch & CUDA 检查")
print("-" * 60)
try:
    import torch
    print(f"[OK] PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   [OK] GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"   [OK] GPU数量: {torch.cuda.device_count()}")
        print(f"   [OK] GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   [OK] CUDA版本: {torch.version.cuda}")
        print(f"   [OK] cuDNN版本: {torch.backends.cudnn.version()}")
    else:
        print("   [INFO] CUDA不可用 - N-HiTS和Chronos将使用CPU（速度较慢）")
except ImportError:
    print("[ERROR] PyTorch未安装")
    print("   安装命令: pip install torch --index-url https://download.pytorch.org/whl/cu130")

# 2. 检查LightGBM
print("\n2. LightGBM 检查")
print("-" * 60)
try:
    import lightgbm as lgb
    print(f"[OK] LightGBM版本: {lgb.__version__}")

    # 尝试创建GPU数据集
    try:
        import numpy as np
        test_data = np.random.rand(100, 10)
        test_label = np.random.rand(100)
        dataset = lgb.Dataset(test_data, label=test_label)

        params = {'device': 'gpu', 'verbose': -1}
        model = lgb.train(params, dataset, num_boost_round=1)
        print("   [OK] GPU支持: 已启用")
        print("   [OK] LightGBM可使用GPU加速")
    except Exception as e:
        print(f"   [INFO] GPU支持: 未启用")
        print(f"   原因: {str(e)}")
        print("   LightGBM将使用CPU")
        print("   安装GPU版本: pip install lightgbm --config-settings=cmake.define.USE_GPU=ON")
except ImportError:
    print("[ERROR] LightGBM未安装")
    print("   安装命令: pip install lightgbm")

# 3. 检查TBATS
print("\n3. TBATS 检查")
print("-" * 60)
try:
    from tbats import TBATS
    print("[OK] TBATS已安装")
    print("   [INFO] TBATS是传统统计模型，不支持GPU")
    print("   [INFO] 使用多进程并行加速（CPU多核）")

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"   [OK] CPU核心数: {cpu_count}")
    print(f"   [OK] 可用n_jobs=-1启用全部{cpu_count}个核心并行")
except ImportError:
    print("[ERROR] TBATS未安装")
    print("   安装命令: pip install tbats")

# 4. 检查Chronos
print("\n4. Chronos-T5 检查")
print("-" * 60)
try:
    from chronos import ChronosPipeline
    print("[OK] Chronos已安装")

    if torch.cuda.is_available():
        print("   [OK] 将使用GPU加速（bfloat16精度）")
        print("   [OK] 推理速度可提升5-20倍")
    else:
        print("   [INFO] 将使用CPU（推理速度较慢）")
except ImportError:
    print("[ERROR] Chronos未安装")
    print("   安装命令: pip install git+https://github.com/amazon-science/chronos-forecasting.git")

# 5. 总结和建议
print("\n" + "=" * 60)
print("总结和建议")
print("=" * 60)

if torch.cuda.is_available():
    print("[OK] GPU环境配置良好!")
    print("\n推荐配置:")
    print("  - TBATS: n_jobs=-1（使用所有CPU核心）")
    print("  - LightGBM: device='auto'（自动检测GPU）")
    print("  - N-HiTS: device='auto', batch_size=64, hidden_size=512")
    print("  - Chronos: device='auto', model='chronos-t5-small'")

    # 根据GPU显存给出建议
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n根据GPU显存 ({gpu_memory:.1f} GB) 的建议:")

    if gpu_memory < 6:
        print("  - N-HiTS: batch_size=16-32, hidden_size=256")
        print("  - Chronos: chronos-t5-tiny 或 mini")
    elif gpu_memory < 12:
        print("  - N-HiTS: batch_size=32-64, hidden_size=512")
        print("  - Chronos: chronos-t5-small")
    elif gpu_memory < 20:
        print("  - N-HiTS: batch_size=64-128, hidden_size=512-1024")
        print("  - Chronos: chronos-t5-base")
    else:
        print("  - N-HiTS: batch_size=128+, hidden_size=1024")
        print("  - Chronos: chronos-t5-large")
else:
    print("[INFO] 未检测到GPU")
    print("\n当前配置（CPU模式）:")
    print("  - TBATS: n_jobs=-1（多进程加速）[推荐]")
    print("  - LightGBM: device='cpu'（速度尚可）")
    print("  - N-HiTS: device='cpu'（训练较慢）[注意]")
    print("  - Chronos: device='cpu'（推理较慢）[注意]")

    print("\n建议:")
    print("  1. 优先使用 TBATS 和 LightGBM（CPU性能尚可）")
    print("  2. 如需使用深度学习模型，建议配置GPU环境")
    print("  3. 安装CUDA和cuDNN: https://pytorch.org/get-started/locally/")

print("\n" + "=" * 60)
print("检查完成!")
print("=" * 60)
