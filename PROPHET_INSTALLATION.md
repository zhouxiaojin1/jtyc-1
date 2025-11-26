# Prophet模型安装与使用说明

## 1. 安装Prophet

### 方法1：使用pip安装（推荐）

```bash
pip install prophet
```

### 方法2：使用conda安装（更稳定）

```bash
conda install -c conda-forge prophet
```

### 依赖库
Prophet需要以下依赖（通常会自动安装）：
- cmdstanpy >= 1.0.4
- numpy >= 1.15.4
- pandas >= 1.0.4
- matplotlib >= 2.0.0
- holidays >= 0.25
- tqdm >= 4.36.1

**注意：**
- Windows系统首次安装Prophet可能需要较长时间（约12MB下载）
- 如果网络较慢，建议使用conda安装或配置pip国内镜像源

### 配置pip国内镜像（可选）

如果下载速度很慢，可以使用清华镜像源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install prophet
```

或者临时使用：

```bash
pip install prophet -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. Prophet vs TBATS 对比

### 性能对比

| 指标 | Prophet | TBATS |
|-----|---------|-------|
| **训练速度** | ⭐⭐⭐⭐⭐ 非常快 | ⭐⭐ 很慢 |
| **预测精度** | ⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中等 |
| **可扩展性** | ⭐⭐⭐⭐⭐ 优秀（并行训练） | ⭐⭐ 差（难以扩展） |
| **可解释性** | ⭐⭐⭐⭐⭐ 强（趋势+季节性分解） | ⭐⭐⭐ 中等 |
| **内存占用** | ⭐⭐⭐⭐ 低 | ⭐⭐⭐ 中等 |
| **易用性** | ⭐⭐⭐⭐⭐ 简单 | ⭐⭐⭐ 复杂 |

### 速度对比（训练100个区域）

- **Prophet**: 约5-15分钟（并行训练）
- **TBATS**: 约1-3小时（并行训练）

**Prophet速度快10-100倍！**

### 特性对比

#### Prophet优势
✅ **自动趋势变化点检测** - 无需手动指定趋势变化
✅ **多季节性支持** - 日、周、年周期自动建模
✅ **节假日效应** - 可以添加节假日和特殊事件
✅ **缺失值处理** - 自动处理缺失数据
✅ **不确定性区间** - 提供预测区间（置信度）
✅ **并行训练** - 多个区域可以并行训练
✅ **可视化分析** - 内置趋势和季节性分解图
✅ **参数直观** - 参数含义清晰，易于调整

#### TBATS优势
✅ **Box-Cox变换** - 自动处理非平稳性
✅ **ARMA误差建模** - 捕捉残差自相关
✅ **理论完备** - 基于经典统计理论

#### TBATS劣势
❌ **训练极慢** - 大规模数据难以应用
❌ **内存占用大** - 长序列内存需求高
❌ **难以并行** - 内部优化难以利用多核
❌ **参数复杂** - 季节性周期需要精确指定
❌ **扩展性差** - 区域数量增加时速度急剧下降

## 3. 使用Prophet替代TBATS

### 在前端界面使用

1. **启动应用**
   ```bash
   streamlit run app.py
   ```

2. **选择模型**
   - 进入"模型训练"页面
   - 选择 **Prophet** 模型（而不是TBATS）

3. **配置参数**（推荐设置）
   - **季节性模式**: multiplicative（适合交通流量）
   - **日周期**: ✅ 开启
   - **周周期**: ✅ 开启
   - **年周期**: ❌ 关闭（除非有1年以上数据）
   - **趋势灵活度**: 0.05（中等）
   - **季节性灵活度**: 10.0（较高）
   - **并行进程数**: 2-4（根据CPU核心数）

4. **训练和预测**
   - 点击"开始训练"
   - 等待训练完成（比TBATS快得多）
   - 查看预测结果和评估指标

### 代码使用示例

```python
from predict.model_prophet import ProphetForecaster
import pandas as pd

# 加载数据
df = pd.read_csv('dataset/milano_traffic_nid.csv')

# 划分训练集和测试集
split_idx = int(len(df) * 0.9)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# 创建Prophet预测器
forecaster = ProphetForecaster(
    forecast_horizon=1008,              # 预测1周（1008个10分钟）
    seasonality_mode='multiplicative',  # 乘法季节性
    daily_seasonality=True,             # 日周期
    weekly_seasonality=True,            # 周周期
    yearly_seasonality=False,           # 年周期（数据不足1年时关闭）
    changepoint_prior_scale=0.05,       # 趋势灵活度
    seasonality_prior_scale=10.0,       # 季节性灵活度
    n_jobs=2                            # 并行训练
)

# 训练
time_col = df.columns[0]
value_cols = df.columns[1:11].tolist()  # 选择前10个区域

train_results = forecaster.fit(
    train_df[[time_col] + value_cols],
    time_col,
    value_cols,
    start_time='2013-11-01 00:00:00',
    parallel=True
)

# 预测
last_train_time = pd.Timestamp('2013-11-01 00:00:00') + pd.Timedelta(minutes=10 * (len(train_df) - 1))
predictions = forecaster.predict(
    value_cols,
    last_train_time=last_train_time,
    steps=1008
)

# 评估
metrics = forecaster.evaluate(
    test_df[[time_col] + value_cols],
    predictions,
    time_col
)

# 保存模型
forecaster.save_model('output/prophet_model.pkl')
```

## 4. Prophet参数调优建议

### 季节性模式（seasonality_mode）

- **multiplicative**（乘法）
  - 适合：交通流量、销售额、用电量等变化幅度随水平变化的数据
  - 特点：季节性波动与水平成正比

- **additive**（加法）
  - 适合：温度、降雨量等变化幅度相对固定的数据
  - 特点：季节性波动与水平无关

### 趋势灵活度（changepoint_prior_scale）

- **0.001-0.01**: 趋势变化少（平滑）
- **0.05**: 默认值，适合大多数情况
- **0.1-0.5**: 趋势变化多（灵活），适合突变多的数据

### 季节性灵活度（seasonality_prior_scale）

- **0.01-1.0**: 季节性较弱
- **10.0**: 默认值，适合强季节性数据（如交通流量）
- **20.0-100.0**: 季节性非常强

### 并行训练（n_jobs）

- 建议设置为 **CPU核心数的1/2到3/4**
- 例如：8核CPU设置为4-6
- 可以显著加速多区域训练

## 5. 输出文件说明

Prophet模型训练完成后会生成以下文件（保存在 `output/` 目录）：

- **prophet_predictions.csv** - 预测结果（每列一个区域）
- **prophet_test_metrics.csv** - 评估指标（MAE、RMSE、MAPE）
- **prophet_predictions_plot.png** - 预测结果可视化
- **prophet_training_summary.png** - 训练汇总图表
- **prophet_model.pkl** - 训练好的模型（可加载复用）

## 6. 常见问题

### Q1: Prophet训练速度还是很慢怎么办？

**A**:
1. 增加 `n_jobs` 参数（并行训练）
2. 减少区域数量（选择部分区域训练）
3. 缩短训练数据长度
4. 设置 `uncertainty_samples=0`（已默认设置，加速预测）

### Q2: 预测精度不理想怎么办？

**A**:
1. 调整 `changepoint_prior_scale`（增大或减小）
2. 调整 `seasonality_prior_scale`（增大以增强季节性）
3. 增加训练数据长度（建议至少2周数据）
4. 尝试切换 `seasonality_mode` (multiplicative ↔ additive)
5. 检查数据质量（缺失值、异常值）

### Q3: 如何加载已保存的模型进行预测？

**A**:
```python
forecaster = ProphetForecaster()
forecaster.load_model('output/prophet_model.pkl')

# 直接预测
predictions = forecaster.predict(
    value_cols,
    last_train_time,
    steps=1008
)
```

### Q4: 如何添加节假日效应？

**A**:
在 `_create_prophet_model()` 方法中添加：
```python
holidays_df = pd.DataFrame({
    'holiday': ['春节', '国庆'],
    'ds': pd.to_datetime(['2013-02-10', '2013-10-01']),
    'lower_window': 0,
    'upper_window': 1,
})
model = Prophet(holidays=holidays_df)
```

## 7. 总结

### 为什么选择Prophet而不是TBATS？

1. **速度快** - 10-100倍的速度优势
2. **效果好** - 自动趋势检测和季节性建模
3. **易用性** - 参数直观，调试简单
4. **可扩展** - 并行训练，支持大规模应用
5. **可解释** - 提供趋势和季节性分解
6. **成熟稳定** - Facebook开源，广泛应用

### Prophet适用场景

✅ 多个区域需要独立预测
✅ 数据有明显的季节性模式
✅ 需要快速训练和部署
✅ 需要可解释的预测结果
✅ 有趋势变化（如增长或下降）

### 不适用场景

❌ 需要利用区域间关联（考虑使用LSTM或LightGBM）
❌ 数据量极少（<100个数据点）
❌ 需要多变量建模

---

**建议：优先使用Prophet替代TBATS进行交通流量预测！**
