# -*- coding: utf-8 -*-

"""
A股行业板块轮动 - 滚动相关性分析模块 (Rolling Correlation)
对应文档章节：5.3.1 计算方法 | 5.3.2 分析结果 | 5.3.3 应用价值
功能：计算并绘制 新能源 vs 半导体 的60日动态相关性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==============================================================================
# 1. 初始化与数据加载
# ==============================================================================

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid', {'font.sans-serif': ['SimHei', 'Microsoft YaHei']})

# 读取之前缓存的数据 (sector_data_cache.csv)
DATA_FILE = 'data/sector_data_cache.csv'

if not os.path.exists(DATA_FILE):
    print(f"错误：找不到数据文件 {DATA_FILE}。请先运行第2步的数据获取代码。")
    sys.exit(1)

print(">>> 正在读取本地数据...")
close_df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# 确保列名对应 (电力设备=新能源, 电子=半导体)
# 如果您的缓存文件中列名已经是中文，直接使用即可
# 映射检查：
target_pair = {
    '新能源': '电力设备', # 文档叫新能源，数据里叫电力设备
    '半导体': '电子'     # 文档叫半导体，数据里叫电子
}

# 检查列是否存在
available_cols = close_df.columns
col_1 = target_pair['新能源'] if target_pair['新能源'] in available_cols else '新能源'
col_2 = target_pair['半导体'] if target_pair['半导体'] in available_cols else '半导体'

if col_1 not in available_cols or col_2 not in available_cols:
    print(f"错误：数据中找不到 {col_1} 或 {col_2} 列。现有列：{list(available_cols)}")
    sys.exit(1)

print(f">>> 正在分析目标对：{col_1} (新能源) vs {col_2} (半导体)")

# ==============================================================================
# 2. 计算滚动相关性 (对应 5.3.1)
# ==============================================================================

# 1. 计算日收益率
returns = np.log(close_df / close_df.shift(1)).dropna()

# 2. 设置窗口大小 (60交易日 ≈ 3个月)
WINDOW_SIZE = 60

# 3. 计算滚动相关系数
# rolling(60).corr() 会计算动态窗口内的皮尔逊相关系数
rolling_corr = returns[col_1].rolling(window=WINDOW_SIZE).corr(returns[col_2])

# 去除起始的空值
rolling_corr = rolling_corr.dropna()

# ==============================================================================
# 3. 可视化分析 (对应 5.3.2)
# ==============================================================================

print(">>> 正在绘制滚动相关性走势图...")

plt.figure(figsize=(14, 6))

# 绘制面积图
plt.fill_between(rolling_corr.index, rolling_corr, 0, alpha=0.3, color='#1f77b4')
plt.plot(rolling_corr.index, rolling_corr, color='#1f77b4', linewidth=1.5, label='60日滚动相关系数')

# 添加辅助线
plt.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='强相关界限 (0.8)')
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='中等相关 (0.5)')
plt.axhline(0, color='black', linewidth=1)

# 装饰图表
plt.title(f'新能源(电力设备) 与 半导体(电子) 板块滚动相关性分析 (2022-2025)\n窗口大小={WINDOW_SIZE}日', 
          fontsize=14, fontweight='bold')
plt.ylabel('相关系数 (Correlation)')
plt.legend(loc='lower right')
plt.ylim(-0.2, 1.0) # 相关系数通常在-1到1之间，这里主要展示正相关区域

# 标记年份区间 (为了对应文中的2022-2025分析)
years = sorted(list(set(rolling_corr.index.year)))
for year in years:
    # 在每年的中间位置标注年份
    mid_date = pd.Timestamp(f"{year}-07-01")
    if mid_date in rolling_corr.index or (mid_date > rolling_corr.index[0] and mid_date < rolling_corr.index[-1]):
        plt.text(mid_date, 0.9, str(year), ha='center', fontsize=20, alpha=0.15, fontweight='bold')

plt.tight_layout()
save_name = 'images/7_滚动相关性分析.png'
plt.savefig(save_name, dpi=300)
print(f">>> 图表已保存为: {save_name}")

# ==============================================================================
# 4. 生成统计报告 (验证 5.3.2 中的结论)
# ==============================================================================

def print_beautiful_table(df, title="表格", index_name=""):
    """
    美化表格打印 - 通用版本
    
    Args:
        df: 要打印的DataFrame
        title: 表格标题
        index_name: 索引列的名称
    """
    # 创建包含索引的临时DataFrame用于宽度计算
    if index_name:
        temp_df = df.copy()
        temp_df[index_name] = df.index
        temp_df = temp_df[[index_name] + list(df.columns)]
    else:
        temp_df = df.reset_index()
    
    # 计算每列的最大宽度
    col_widths = {}
    for col in temp_df.columns:
        max_width = max(len(str(x)) for x in temp_df[col])
        col_widths[col] = max(max_width, len(str(col)))
    
    # 表头
    header = " | ".join(str(col).ljust(col_widths[col]) for col in temp_df.columns)
    separator = "-+-".join("-" * col_widths[col] for col in temp_df.columns)
    
    # 计算标题居中位置
    title_padding = (len(header) - len(title)) // 2
    title_line = " " * title_padding + title
    
    print("\n" + "="*len(header))
    print(title_line)
    print("="*len(header))
    print(header)
    print(separator)
    
    # 数据行
    for _, row in temp_df.iterrows():
        row_str = " | ".join(str(row[col]).ljust(col_widths[col]) for col in temp_df.columns)
        print(row_str)
    
    print("="*len(header))

# 按年份分组计算相关系数的均值、最大值、最小值
yearly_stats = rolling_corr.groupby(rolling_corr.index.year).agg(['mean', 'min', 'max'])
yearly_stats.columns = ['平均相关性', '最低', '最高']

# 格式化数据
yearly_stats_formatted = yearly_stats.map(lambda x: f"{x:.2f}")

# 美化打印
print_beautiful_table(yearly_stats_formatted, title="5.3.2 滚动相关性年度统计表", index_name="年份")