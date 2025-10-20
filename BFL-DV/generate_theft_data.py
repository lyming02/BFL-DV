import os
import pandas as pd
import numpy as np
import random
from glob import glob
from tqdm import tqdm
import torch  # 新增

# 数据目录
DATA_DIR = 'data/Small LCL Data'
OUTPUT_CSV = 'theft_dataset.csv'

# 读取所有csv文件
csv_files = glob(os.path.join(DATA_DIR, '*.csv'))

all_weeks = []
user_week_index = []  # 记录每个周属于哪个用户、哪一周

for file in tqdm(csv_files, desc='处理用户csv文件'):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # 去除列名空格
    # 自动查找用电量列
    kwh_col = None
    for col in df.columns:
        if 'KWH' in col and 'hh' in col:
            kwh_col = col
            break
    if kwh_col is None:
        raise ValueError(f"文件{file}未找到用电量列，实际列名为：{df.columns.tolist()}")
    # 只保留需要的列
    df = df[['LCLid', 'DateTime', kwh_col]]
    # 补全时间索引
    df = df.reset_index(drop=True)
    # 按半小时为单位，每周336条
    num_weeks = len(df) // 336
    # 用电量列转为数值，'Null'等无效值转为0
    df[kwh_col] = pd.to_numeric(df[kwh_col], errors='coerce').fillna(0)
    for i in range(num_weeks):
        week_df = df.iloc[i*336:(i+1)*336].copy()
        if len(week_df) == 336:
            all_weeks.append(week_df)
            user_week_index.append((week_df['LCLid'].iloc[0], i))

# 合并所有周为一个大DataFrame，每周为一个实例
X = np.array([week[kwh_col].values for week in all_weeks])
X = pd.DataFrame(X)
X['LCLid'] = [uw[0] for uw in user_week_index]
X['week_index'] = [uw[1] for uw in user_week_index]
X['label'] = 0  # 默认正常

# 选取15%周实例注入窃电
num_theft = int(0.85 * len(X))
theft_indices = np.random.choice(X.index, num_theft, replace=False)

# 生成"正常水平"窃电：整体缩减10%~30%，并只对部分时段（如夜间/白天）缩减
week_data = X.iloc[theft_indices, :336].values.astype(np.float32)
week_tensor = torch.tensor(week_data)

for i in range(len(theft_indices)):
    # 1. 整体缩减
    alpha = np.random.uniform(0.7, 0.9)  # 只缩减10%~30%
    week_tensor[i] = week_tensor[i] * alpha

    # 2. 只对部分时段缩减（如夜间0~96，白天96~240，晚间240~336）
    if np.random.rand() < 0.5:
        # 只缩减夜间
        week_tensor[i, 0:96] *= np.random.uniform(0.5, 0.8)
    else:
        # 只缩减白天
        week_tensor[i, 96:240] *= np.random.uniform(0.5, 0.8)

# 写回DataFrame
X.iloc[theft_indices, :336] = week_tensor.numpy()
X.loc[theft_indices, 'label'] = 1
X.loc[theft_indices, 'FDI_type'] = 'normal_level'

# 保存
X.to_csv(OUTPUT_CSV, index=False)
print(f'已生成窃电数据集：{OUTPUT_CSV}') 