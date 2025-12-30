# -*- coding: utf-8 -*-

"""
A股行业板块轮动分析系统 (季度增强版)
功能：Tushare Pro 数据获取 | 自动重试 | 本地缓存 | 全历史季度热力图
"""

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import sys
import os
import time

# ==============================================================================
# 1. 初始化设置
# ==============================================================================

def load_token_from_env():
    env_file = '.env'
    if not os.path.exists(env_file):
        print(f"!!! 错误：找不到 {env_file} 文件")
        sys.exit(1)
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key.strip() == 'TUSHARE_TOKEN':
                        return value.strip().strip("'").strip('"')
        print("!!! 错误：在 .env 文件中找不到 TUSHARE_TOKEN")
        sys.exit(1)
    except Exception as e:
        print(f"!!! 读取 .env 文件失败: {e}")
        sys.exit(1)

MY_TOKEN = load_token_from_env()

# 绘图与打印设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("notebook", font_scale=1.1)
sns.set_style('whitegrid', {'font.sans-serif': ['SimHei', 'Microsoft YaHei']})
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    ts.set_token(MY_TOKEN)
    pro = ts.pro_api()
    print(">>> Tushare 接口初始化成功")
except Exception as e:
    print(f"!!! 初始化失败: {e}")
    sys.exit(1)

# ==============================================================================
# 2. 数据获取
# ==============================================================================

DATA_FILE = 'data/sector_data_cache.csv'

SECTORS_MAP = {
    '食品饮料': '801120.SI',
    '电力设备': '801730.SI', # 新能源
    '电子':     '801080.SI', # 半导体
    '医药生物': '801150.SI',
    '有色金属': '801050.SI',
    '银行':     '801780.SI'
}

def get_data_smart():
    # 简单的本地缓存逻辑
    if os.path.exists(DATA_FILE):
        try:
            print(f">>> 读取本地缓存: {DATA_FILE}")
            return pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        except:
            pass
            
    print(f">>> [联网模式] 正在下载数据...")
    start_date = '20220101'
    end_date = '20251226'
    data_dict = {}
    
    for name, code in SECTORS_MAP.items():
        try:
            df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date)
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                data_dict[name] = df['close']
                print(f"  - {name} 下载成功")
            else:
                print(f"  - {name} 无数据")
        except Exception as e:
            print(f"  - {name} 失败: {e}")
            
    if data_dict:
        full_df = pd.DataFrame(data_dict).ffill().dropna()
        full_df.to_csv(DATA_FILE, encoding='utf-8-sig')
        return full_df
    return pd.DataFrame()

close_df = get_data_smart()
if close_df.empty:
    print("无数据，退出")
    sys.exit(1)

return_df = np.log(close_df / close_df.shift(1)).dropna()

# ==============================================================================
# 3. 指标计算
# ==============================================================================

# 基础指标
ann_ret = np.exp(return_df.mean() * 252) - 1
total_ret = np.exp(return_df.sum()) - 1
ann_vol = return_df.std() * np.sqrt(252)
sharpe = (ann_ret - 0.028) / ann_vol

# 季度收益率计算
q_ret = return_df.resample('Q').apply(lambda x: np.exp(x.sum()) - 1)
# 格式化索引为 2022Q1 格式
q_ret.index = [f"{x.year}Q{x.quarter}" for x in q_ret.index]

# ==============================================================================
# 4. 可视化绘制 (增强版)
# ==============================================================================

print("\n>>> 正在绘制增强版图表...")

# 图1: 累计收益 (保持不变)
plt.figure(figsize=(14, 7))
norm_close = close_df / close_df.iloc[0]
for col in norm_close.columns:
    plt.plot(norm_close.index, norm_close[col], lw=2, alpha=0.9, label=col)
plt.title('A股行业板块累计收益率走势', fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/1_累计收益率.png', dpi=300)

# 图2: 风险收益 (保持不变)
plt.figure(figsize=(10, 6))
plt.scatter(ann_vol*100, ann_ret*100, s=(sharpe+1)*500, c=sharpe, cmap='RdYlGn', edgecolors='k', alpha=0.8)
for i, txt in enumerate(ann_ret.index):
    plt.annotate(txt, (ann_vol[i]*100, ann_ret[i]*100), ha='center', va='center', fontweight='bold')
plt.xlabel('年化波动率(%)'); plt.ylabel('年化收益率(%)')
plt.title('风险收益气泡图')
plt.tight_layout()
plt.savefig('images/2_风险收益.png', dpi=300)

# ---【核心修改】图4: 全历史季度热力图 (The Big Picture) ---
plt.figure(figsize=(16, 8))
# 转置：行是板块，列是季度
heatmap_data = q_ret.T * 100 

# 自定义红绿配色 (中国习惯: 红涨绿跌)
# diverging_palette: h_neg=145(绿), h_pos=10(红)
cmap = sns.diverging_palette(145, 10, as_cmap=True, s=90, l=50)

ax = sns.heatmap(heatmap_data, 
            annot=True, 
            fmt='.1f', 
            cmap=cmap, 
            center=0,
            linewidths=1, 
            linecolor='white',
            cbar_kws={'label': '季度收益率 (%)'})

# 可以在格子里加百分号
for t in ax.texts: t.set_text(t.get_text() + "%")

plt.title('A股行业板块 - 全历史季度轮动热力图 (2022-2025)', fontsize=18, pad=20)
plt.xlabel('季度', fontsize=12)
plt.ylabel('行业板块', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('images/3_季度轮动_全景热力图.png', dpi=300)
print(">>> [新增] 全景热力图已保存: images/3_季度轮动_全景热力图.png")

# ---【核心修改】图5: 最近 9 个季度详情 (3x3 Grid) ---
# 获取最近 9 个季度
recent_q = q_ret.iloc[-9:] 
num_plots = len(recent_q)
rows, cols = 3, 3 # 3行3列

fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
axes = axes.flatten()

for i in range(rows * cols):
    if i < num_plots:
        # 倒序排列：最新的在右下角，或者按时间正序。这里按时间正序
        date = recent_q.index[i]
        row_data = recent_q.iloc[i].sort_values(ascending=False)
        
        ax = axes[i]
        colors = ['#d62728' if v >= 0 else '#2ca02c' for v in row_data.values]
        
        # 画横向柱状图可能省空间，这里维持纵向
        rects = ax.bar(row_data.index, row_data.values * 100, color=colors, alpha=0.85)
        
        ax.set_title(f"{date} 表现", fontsize=12, fontweight='bold', pad=10)
        ax.axhline(0, color='black', lw=0.8)
        ax.grid(axis='y', ls='--', alpha=0.3)
        ax.set_ylim(min(recent_q.min().min()*100, -10), max(recent_q.max().max()*100, 10)) # 统一Y轴刻度视觉更好
        
        # 简化标签，防止重叠
        ax.tick_params(axis='x', labelsize=9)
        
        # 标注数值
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + (1 if h>0 else -3), 
                    f"{h:.0f}", ha='center', fontsize=9, color='black')
    else:
        axes[i].axis('off') # 隐藏多余的子图

plt.suptitle(f'最近 {num_plots} 个季度行业表现详情', fontsize=20, y=0.98)
plt.tight_layout()
plt.savefig('images/4_季度轮动_最近9季详情.png', dpi=300)
print(">>> [新增] 详情网格图已保存: images/4_季度轮动_最近9季详情.png")

# ==============================================================================
# 5. 打印完整报告
# ==============================================================================

print("\n" + "="*80)
print("             季度轮动深度分析数据")
print("="*80)

# 1. 完整季度数据打印
print(f"\n[1. 历史季度收益率总表 (共 {len(q_ret)} 个季度)]")
# 格式化为百分比字符串显示
pd.set_option('display.max_rows', None)
print(q_ret.applymap(lambda x: f"{x:.2%}").iloc[::-1]) # 倒序打印，最新的在上面

# 2. 季度冠军统计
print(f"\n[2. 季度领涨板块统计]")
winners = q_ret.idxmax(axis=1)
winners_val = q_ret.max(axis=1)
summary = pd.DataFrame({'领涨板块': winners, '涨幅': winners_val})
summary['涨幅'] = summary['涨幅'].map(lambda x: f"{x:.2%}")
print(summary.iloc[::-1].head(9)) # 打印最近9个季度的冠军

# 3. 导出
q_ret.to_csv('data/完整季度收益率.csv', encoding='utf-8-sig')
print("\n>>> 完整数据已导出至: data/完整季度收益率.csv")