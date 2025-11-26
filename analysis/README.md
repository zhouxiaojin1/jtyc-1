# 聚类分析模块
##测试.................................
## 功能简介

时间序列聚类分析模块，用于发现具有相似流量模式的区域。

## 支持的聚类方法

### 1. K-Means 聚类
- **优点**: 快速、适合大规模数据
- **适用场景**: 初步分析、快速分组
- **参数**: 聚类数 k

### 2. 层次聚类 (Hierarchical Clustering)
- **优点**: 可生成树状图、层次关系清晰
- **适用场景**: 需要理解层次结构时
- **参数**: 聚类数 k、连接方式

### 3. K-Shape 聚类
- **优点**: 专门针对时间序列、基于形状相似度
- **适用场景**: 时间序列模式分析
- **参数**: 聚类数 k

## 使用方法

### 命令行使用

```bash
cd F:\jtyc\new
python analysis/clustering_analysis.py
```

### 代码使用

```python
from analysis.clustering_analysis import TimeSeriesClustering
import pandas as pd

# 加载数据
df = pd.read_csv('dataset/milano_traffic_nid.csv')

# 创建聚类器
clustering = TimeSeriesClustering(
    n_clusters=5,
    method='kmeans',
    normalize=True
)

# 执行聚类
labels = clustering.fit(df, time_col=df.columns[0])

# 可视化
clustering.plot_cluster_centers(save_path='cluster_centers.png')
clustering.plot_pca_visualization(df, time_col=df.columns[0])

# 寻找最优k
fig, best_k = clustering.find_optimal_k(df, k_range=range(2, 11))
```

### Web界面使用

1. 启动应用:
   ```bash
   streamlit run app.py
   ```

2. 在侧边栏选择"聚类分析"

3. 按照以下步骤操作:
   - **数据准备**: 加载数据集，选择区域和时间范围
   - **聚类分析**: 选择聚类方法，设置参数，执行聚类
   - **结果查看**: 查看聚类中心、分布、PCA可视化等

## 评估指标

### 1. 轮廓系数 (Silhouette Score)
- **范围**: [-1, 1]
- **含义**: 越接近1表示聚类效果越好
- **计算**: 衡量样本与其所在聚类的相似度与其他聚类的差异度

### 2. Davies-Bouldin 指数
- **范围**: [0, ∞)
- **含义**: 越小表示聚类效果越好
- **计算**: 聚类内距离与聚类间距离的比值

### 3. Calinski-Harabasz 指数
- **范围**: [0, ∞)
- **含义**: 越大表示聚类效果越好
- **计算**: 聚类间方差与聚类内方差的比值

## 输出文件

聚类分析结果保存在 `output/clustering/` 目录下:

- `cluster_result.csv`: 每个区域的聚类标签
- `cluster_summary.csv`: 聚类摘要统计
- `cluster_centers.png`: 聚类中心曲线
- `cluster_distribution.png`: 聚类分布图
- `pca_visualization.png`: PCA降维可视化
- `dendrogram.png`: 层次聚类树状图
- `optimal_k.png`: 最优k值分析

## 应用场景

1. **区域分组**: 将具有相似流量模式的区域分组
2. **异常检测**: 识别流量模式异常的区域
3. **资源分配**: 根据流量模式优化资源分配
4. **预测优化**: 为不同模式的区域使用不同的预测模型

## 技术细节

- **数据标准化**: 使用 StandardScaler 进行 Z-score 标准化
- **距离度量**:
  - K-Means: 欧氏距离
  - K-Shape: 形状距离（基于互相关）
  - 层次聚类: Ward 连接
- **PCA降维**: 降到2维用于可视化

## 依赖库

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
