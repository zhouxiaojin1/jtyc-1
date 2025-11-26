"""
测试中文字体显示
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.plot_config import setup_chinese_font, ensure_chinese_font

# 设置中文字体
setup_chinese_font()

# 创建测试图表
fig, ax = plt.subplots(figsize=(10, 6))

# 确保中文字体生效
ensure_chinese_font()

# 生成测试数据
x = np.arange(0, 100)
y1 = np.sin(x / 10) * 50 + 100
y2 = np.cos(x / 10) * 30 + 100
y3 = np.sin(x / 15) * 40 + 100

# 绘制曲线
ax.plot(x, y1, label='训练数据', color='blue', linewidth=2)
ax.plot(x, y2, label='真实值', color='green', linewidth=2)
ax.plot(x, y3, label='预测值', color='red', linestyle='--', linewidth=2)

# 添加垂直分界线
ax.axvline(x=70, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# 设置标题和标签
ax.set_title('区域: 测试区域 - 中文字体显示测试', fontsize=14, fontweight='bold')
ax.set_xlabel('时间步 (10分钟间隔)', fontsize=12)
ax.set_ylabel('交通流量', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# 保存图片
output_path = 'output/test_chinese_font.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"测试图片已保存到: {output_path}")
print("请打开图片检查中文是否正确显示")

plt.show()
plt.close()
