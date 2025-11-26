"""
环境诊断脚本
用于验证subprocess调用时的环境配置
"""

import subprocess
import sys
import os

print("=" * 60)
print("环境诊断报告")
print("=" * 60)

print(f"\n1. 当前Python路径:")
print(f"   {sys.executable}")

print(f"\n2. Python版本:")
print(f"   {sys.version}")

print(f"\n3. 关键包检查:")
packages = [
    'numpy', 'pandas', 'torch', 'lightgbm', 'tbats',
    'sklearn', 'matplotlib', 'seaborn', 'tqdm', 'streamlit'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'N/A')
        print(f"   [{pkg:15s}] OK - version {ver}")
    except ImportError:
        print(f"   [{pkg:15s}] MISSING")

print(f"\n4. 测试subprocess调用:")

# 创建临时测试脚本
test_script_content = """
import sys
print("Subprocess Python:", sys.executable)

errors = []
packages = ['torch', 'lightgbm', 'tbats', 'pandas', 'numpy']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  Import {pkg}: OK")
    except ImportError as e:
        print(f"  Import {pkg}: FAILED - {e}")
        errors.append(pkg)

if errors:
    print("FAILED packages:", errors)
    sys.exit(1)
else:
    print("All imports successful!")
    sys.exit(0)
"""

temp_file = "temp_subprocess_test.py"
with open(temp_file, "w", encoding="utf-8") as f:
    f.write(test_script_content)

try:
    # 使用与Streamlit相同的方式调用subprocess
    python_path = sys.executable
    env = os.environ.copy()

    process = subprocess.Popen(
        [python_path, temp_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    output, _ = process.communicate()
    print(output)

    if process.returncode == 0:
        print("\n结论: subprocess调用正常，环境配置正确!")
    else:
        print(f"\n警告: subprocess调用失败，返回码: {process.returncode}")

finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print("=" * 60)
print("诊断完成")
print("=" * 60)
