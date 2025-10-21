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

# 定义FDI类型列表及对应概率（可根据需求调整概率分布）
fdi_types = [
    'FDI1',    # 全时段按比例缩减
    'FDI2',        # 仅夜间时段（0~96）缩减
    'FDI3', # 超过阈值的数值被截断
    'FDI4',   # 整体减去随机常数，下限为0
    'FDI5',   # 按比例缩放
    'FDI6'     # 基于整体均值的比例缩放
]
fdi_probs = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # 定义概率分布

fdi_types_list = []  # 记录每个实例的FDI类型

for i in range(len(theft_indices)):
    # 为每个实例随机选择FDI类型
    selected_type = np.random.choice(fdi_types, p=fdi_probs)
    fdi_types_list.append(selected_type)
    if selected_type == 'FDI1':
        # 整体缩减
        alpha = np.random.uniform(0.2, 0.8)
        week_tensor[i] = week_tensor[i] * alpha
    elif selected_type == 'FDI5':
        alphas = np.random.uniform(0.2, 0.8, size=week_tensor[i].shape)
        week_tensor[i] *= alphas # 逐点相乘（每个点的α不同）
    elif selected_type == 'FDI4':
        # 部分时段缩减
        if np.random.rand() < 0.5:
            week_tensor[i, 0:96] *= np.random.uniform(0.2, 0.8)
        else:
            week_tensor[i, 96:240] *= np.random.uniform(0.2, 0.8)
    elif selected_type == 'FDI2':
        # 截断操作
        mean = week_tensor[i].mean().item()
        std = week_tensor[i].std().item()
        threshold = mean + std
        week_tensor[i] = torch.where(week_tensor[i] <= threshold, week_tensor[i], threshold)
    elif selected_type == 'FDI3':
        # 减去c并取max(·, 0)
        c = np.random.uniform(0, week_tensor[i].max().item())
        week_tensor[i] = torch.maximum(week_tensor[i] - c, torch.tensor(0.0))
    elif selected_type == 'FDI6':
        # 按均值比例缩放
        mean = week_tensor[i].mean().item()
        alpha_t = np.random.uniform(0.2, 0.8)
        week_tensor[i] = torch.full_like(week_tensor[i], alpha_t * mean)

# 写回DataFrame
X.iloc[theft_indices, :336] = week_tensor.numpy()
X.loc[theft_indices, 'label'] = 1
X.loc[theft_indices, 'FDI_type'] = fdi_types_list

# 保存
X.to_csv(OUTPUT_CSV, index=False)
print(f'已生成窃电数据集：{OUTPUT_CSV}') 
