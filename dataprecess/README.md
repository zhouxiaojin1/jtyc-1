# 交通数据预处理模块

本文件夹包含三个数据预处理方法的实现，所有算法优先使用GPU加速。

## 📁 文件说明

### 1. `missing_value_imputation.py`
**缺失值填补：STL + 状态空间卡尔曼平滑器（Seasonal Kalman Smoothing）**

#### 原理
- 先使用STL分解提取季节成分（日季节性=144、周季节性=1008）
- 再在状态空间模型中用卡尔曼滤波/平滑对趋势与残差进行插补

#### 适配性
能同时处理长短季节与突发缺口，插补平滑而不削弱季节结构

#### 关键参数
- `daily_period=144`: 日季节性周期（10分钟间隔）
- `weekly_period=1008`: 周季节性周期（7天）
- `device='cuda'`: 使用GPU加速

#### 使用示例
```python
from missing_value_imputation import STLKalmanImputer
import pandas as pd

# 创建填补器
imputer = STLKalmanImputer(
    daily_period=144,
    weekly_period=1008,
    device='cuda'
)

# 加载数据
df = pd.read_csv('milano_traffic_nid.csv')

# 填补缺失值
df_imputed = imputer.impute_dataframe(df)
```

---

### 2. `anomaly_detection.py`
**异常数据分析：S-H-ESD（Seasonal Hybrid ESD，季节混合广义ESD）**

#### 原理
- 对STL分解的残差部分应用ESD检验
- 基于MAD的稳健Z分数识别异常
- 在保留季节性的前提下识别异常尖峰/跌落

#### 适配性
对强季节交通数据效果稳定，能同时检测双向异常（高/低）

#### 关键参数
- `period=144`: 季节性周期（或周=1008）
- `max_anoms=0.01-0.05`: 最大异常比例（1%-5%）
- `alpha=0.05`: 显著性水平
- `direction='both'`: 检测方向（both/pos/neg）

#### 异常替换方法
对检测到的异常可采用：
- `seasonal_median`: 邻近时段的季节均值替换（推荐）
- `interpolate`: 线性插值
- `stl_reconstruct`: STL重构（趋势+季节）

#### 使用示例
```python
from anomaly_detection import SeasonalHybridESD
import pandas as pd

# 创建检测器
detector = SeasonalHybridESD(
    period=144,
    max_anoms=0.05,
    alpha=0.05,
    direction='both',
    device='cuda'
)

# 加载数据
df = pd.read_csv('milano_traffic_nid.csv')

# 检测并清理异常
df_cleaned, results = detector.detect_and_clean_dataframe(
    df,
    replace_method='seasonal_median'
)
```

---

### 3. `clustering_analysis.py`
**聚类分析：k-Shape（基于形状的时序聚类）**

#### 原理
- 对每个区域的标准化日轮廓进行形状相似度聚类
- 使用z-normalize的24小时×10分钟=144点
- 按模式而非幅度分群

#### 适配性
适合找"早晚高峰形态差异"与"周末模式"，对尺度不敏感

#### 处理流程
1. 提取最近4-8周的工作日/周末日轮廓
2. z-normalize标准化
3. 使用k-Shape聚类
4. k值用轮廓系数/肘部法选择
5. 输出每簇原型曲线

#### 使用示例
```python
from clustering_analysis import TrafficPatternClustering
import pandas as pd

# 创建聚类器
clustering = TrafficPatternClustering(
    daily_points=144,
    n_weeks=4,
    device='cuda'
)

# 加载数据
df = pd.read_csv('milano_traffic_nid.csv')

# 工作日模式聚类
results_weekday, profiles = clustering.cluster_regions(
    df,
    weekday=True,
    auto_k=True  # 自动选择最优k值
)

# 可视化聚类结果
clustering.visualize_clusters(
    results_weekday,
    profiles,
    save_path='weekday_clusters.png'
)
```

---

## 🚀 快速开始

### 环境要求
```bash
pip install numpy pandas torch scipy statsmodels scikit-learn matplotlib seaborn
```

### GPU加速
所有算法都支持GPU加速，会自动检测CUDA是否可用：
- 如果有GPU：自动使用 `device='cuda'`
- 如果无GPU：自动使用 `device='cpu'`

### 完整数据处理流程

```python
import pandas as pd
from missing_value_imputation import STLKalmanImputer
from anomaly_detection import SeasonalHybridESD
from clustering_analysis import TrafficPatternClustering

# 1. 加载数据
df = pd.read_csv('../dataset/milano_traffic_nid.csv')

# 2. 缺失值填补
imputer = STLKalmanImputer(device='cuda')
df_imputed = imputer.impute_dataframe(df)

# 3. 异常检测与清理
detector = SeasonalHybridESD(period=144, device='cuda')
df_cleaned, anomaly_results = detector.detect_and_clean_dataframe(df_imputed)

# 4. 聚类分析
clustering = TrafficPatternClustering(device='cuda')
weekday_results, weekday_profiles = clustering.cluster_regions(
    df_cleaned,
    weekday=True
)
weekend_results, weekend_profiles = clustering.cluster_regions(
    df_cleaned,
    weekday=False
)

# 5. 可视化
clustering.visualize_clusters(weekday_results, weekday_profiles,
                             save_path='weekday_clusters.png')
clustering.visualize_clusters(weekend_results, weekend_profiles,
                             save_path='weekend_clusters.png')
```

---

## 📊 数据格式

### 输入数据格式
CSV文件，第一列为时间戳，其余列为各区域的交通流量：

```
,Region1,Region2,Region3,...
2013-11-01 00:00:00,1834.28,455.64,486.30,...
2013-11-01 00:10:00,1799.29,396.88,396.69,...
...
```

### 输出结果
- **缺失值填补**：填补后的CSV文件
- **异常检测**：清理后的CSV文件 + 检测报告TXT
- **聚类分析**：聚类报告TXT + 可视化PNG图片

---

## ⚙️ 性能优化

### GPU加速
所有算法都使用PyTorch实现，充分利用GPU并行计算能力：
- 卡尔曼滤波的矩阵运算
- SBD距离计算（使用FFT加速）
- 批量数据处理

### 内存优化
- 对大规模数据集，建议分批处理
- 可以先处理部分区域进行测试
- 使用`float32`而非`float64`以节省内存

---

## 📝 注意事项

1. **数据周期性**：算法假设数据具有日/周季节性，适合交通流量等周期性数据
2. **缺失值比例**：缺失值比例过高（>30%）时，填补效果可能下降
3. **异常检测阈值**：根据实际数据调整`max_anoms`和`alpha`参数
4. **聚类数量**：建议使用`auto_k=True`自动选择，或通过轮廓系数评估

---

## 🔧 故障排除

### GPU内存不足
```python
# 减少批次大小或使用CPU
device = 'cpu'
```

### STL分解失败
```python
# 数据太短或缺失太多，算法会自动回退到简单分解
# 建议确保至少有2个完整周期的数据
```

### 聚类结果不理想
```python
# 尝试调整参数
clustering = TrafficPatternClustering(
    daily_points=144,
    n_weeks=8,  # 增加周数
    device='cuda'
)
```

---

## 📚 参考文献

1. **STL**: Cleveland et al. (1990) "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
2. **ESD**: Rosner (1983) "Percentage Points for a Generalized ESD Many-Outlier Procedure"
3. **k-Shape**: Paparrizos & Gravano (2015) "k-Shape: Efficient and Accurate Clustering of Time Series"

---

## 👥 作者
数据预处理模块 - 交通流量预测项目

## 📅 更新日期
2025-01
