"""
绘图配置工具
解决中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import platform
import warnings

warnings.filterwarnings('ignore')


def setup_chinese_font():
    """
    配置matplotlib的中文字体
    自动检测操作系统并设置合适的中文字体
    """
    import matplotlib

    system = platform.system()

    if system == 'Windows':
        # Windows系统字体
        font_list = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',               # 黑体
            'KaiTi',                # 楷体
            'SimSun',               # 宋体
            'FangSong'              # 仿宋
        ]
    elif system == 'Darwin':  # macOS
        font_list = [
            'PingFang SC',          # 苹方-简
            'Heiti SC',             # 黑体-简
            'STHeiti',              # 华文黑体
            'Arial Unicode MS'
        ]
    else:  # Linux
        font_list = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Droid Sans Fallback',
            'AR PL UMing CN',       # 文鼎PL简中明
            'Noto Sans CJK SC'      # 思源黑体
        ]

    # 清除matplotlib的字体缓存
    try:
        matplotlib.font_manager._rebuild()
    except:
        pass

    # 尝试设置字体
    font_set = False
    selected_font = None

    for font_name in font_list:
        try:
            # 检查字体是否在系统中可用
            available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
            if font_name in available_fonts:
                selected_font = font_name
                font_set = True
                print(f"[字体配置] 成功设置中文字体: {font_name}")
                break
        except Exception as e:
            continue

    if font_set and selected_font:
        # 设置字体（确保在列表最前面）
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.family'] = 'sans-serif'
    else:
        # 如果所有字体都不可用，强制使用 SimHei（Windows默认应该有）
        print("[字体配置] 警告: 使用默认中文字体配置")
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'

    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False

    # 其他常用配置
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['figure.figsize'] = (10, 6)
    matplotlib.rcParams['font.size'] = 10

    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # 设置全局样式
    try:
        plt.style.use('default')
    except:
        pass

    # 强制刷新字体设置（在设置样式后再次设置）
    if font_set and selected_font:
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.sans-serif'] = [selected_font]
    else:
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']

    # 打印当前字体配置（用于调试）
    print(f"[字体配置] 当前 font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"[字体配置] 当前 font.family: {plt.rcParams['font.family']}")

    # 返回选定的字体，供其他函数使用
    return selected_font if font_set else 'Microsoft YaHei'


def get_color_palette(n_colors=10):
    """
    获取颜色调色板

    Parameters:
    -----------
    n_colors : int
        需要的颜色数量

    Returns:
    --------
    colors : list
        颜色列表
    """
    import seaborn as sns
    return sns.color_palette("husl", n_colors)


def ensure_chinese_font():
    """
    确保中文字体设置生效（每次绘图前调用）
    """
    import matplotlib
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi']
    plt.rcParams['font.sans-serif'] = chinese_fonts
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['axes.unicode_minus'] = False


def apply_plot_style(ax, title=None, xlabel=None, ylabel=None,
                     grid=True, legend=True):
    """
    应用统一的绘图风格

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        图表对象
    title : str, optional
        标题
    xlabel : str, optional
        x轴标签
    ylabel : str, optional
        y轴标签
    grid : bool
        是否显示网格
    legend : bool
        是否显示图例
    """
    # 确保中文字体生效
    ensure_chinese_font()

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    if legend:
        ax.legend(loc='best', framealpha=0.9, fontsize=9)

    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=9)


# 在导入时自动配置
setup_chinese_font()
