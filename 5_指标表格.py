# -*- coding: utf-8 -*-

"""
A股行业板块轮动分析 - 定制化报表输出版
功能：计算并打印 1.相关性矩阵 2.季度收益表(横向) 3.风险指标 4.收益指标
"""

import tushare as ts
import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# 1. 基础设置与数据获取
# ==============================================================================

# ----------------- 鉴权设置 -----------------
def get_token():
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('TUSHARE_TOKEN'):
                    return line.strip().split('=', 1)[1].strip().strip("'").strip('"')
    return None

MY_TOKEN = get_token()
if not MY_TOKEN:
    print("!!! 错误：请配置 .env 文件或在代码中填入 Token")
    sys.exit(1)

ts.set_token(MY_TOKEN)
pro = ts.pro_api()

# ----------------- 数据显示设置 (关键) -----------------
pd.set_option('display.max_columns', None)      # 显示所有列
pd.set_option('display.max_rows', None)         # 显示所有行
pd.set_option('display.width', 1000)            # 防止换行
pd.set_option('display.unicode.ambiguous_as_wide', True) # 对齐中文
pd.set_option('display.unicode.east_asian_width', True)  # 对齐中文
pd.set_option('display.float_format', lambda x: '%.2f' % x) # 默认保留2位小数

# ----------------- 下载或读取数据 -----------------
SECTORS = {
    '新能源': '801730.SI',
    '半导体': '801080.SI',
    '医药生物': '801150.SI',
    '食品饮料': '801120.SI',
    '有色金属': '801050.SI',
    '银行':     '801780.SI'
}

print(">>> 正在获取数据 (2022-2025)...")
data_dict = {}
for name, code in SECTORS.items():
    try:
        # 尝试读取本地缓存以加快速度
        cache_file = f"data/cache_{name}.csv"
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            df = pro.sw_daily(ts_code=code, start_date='20220101', end_date='20251226')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()
            # 简单缓存
            df.to_csv(cache_file)
            
        data_dict[name] = df['close']
    except:
        print(f"  X {name} 获取失败")

close_df = pd.DataFrame(data_dict).ffill().dropna()

# ==============================================================================
# 2. 核心计算 (生成您需要的四张表)
# ==============================================================================

# 基础收益率
log_ret = np.log(close_df / close_df.shift(1)).dropna() # 对数收益用于累加
pct_ret = close_df.pct_change().dropna()                # 简单收益用于显示

# --- 表1：相关性矩阵 ---
corr_matrix = pct_ret.corr()

# --- 表2：季度收益表 (行=板块，列=季度) ---
# 计算季度收益
q_ret = log_ret.resample('Q').apply(lambda x: np.exp(x.sum()) - 1)
# 格式化列名：2022-03-31 -> 2022Q1
q_ret.index = [f"{x.year}Q{x.quarter}" for x in q_ret.index]
# 转置：让板块变成行，季度变成列 (符合您的要求)
q_ret_table = q_ret.T

# --- 表3：风险指标 (波动率、最大回撤、夏普) ---
# 年化波动率
ann_vol = pct_ret.std() * np.sqrt(252) * 100 
# 最大回撤
def calc_max_dd(series):
    cum = (1 + series).cumprod()
    max_val = cum.cummax()
    dd = (cum - max_val) / max_val
    return dd.min() * 100
max_dd = pct_ret.apply(calc_max_dd)
# 夏普比率 (Rf=2.8%)
ann_ret_simple = (np.power(1 + pct_ret.mean(), 252) - 1) * 100 # 简单年化
sharpe = (ann_ret_simple - 2.8) / ann_vol

risk_table = pd.DataFrame({
    '年化波动率 (%)': ann_vol,
    '最大回撤 (%)': max_dd,
    '夏普比率': sharpe
})

# --- 表4：收益指标 (累计、年化、季度平均) ---
# 累计收益
cum_ret = (np.exp(log_ret.sum()) - 1) * 100
# 年化收益 (复利)
ann_ret_geo = (np.exp(log_ret.mean() * 252) - 1) * 100
# 季度平均收益
q_avg = q_ret.mean() * 100

return_table = pd.DataFrame({
    '累计收益率 (%)': cum_ret,
    '年化收益率 (%)': ann_ret_geo,
    '季度平均收益率 (%)': q_avg
})

# ==============================================================================
# 3. 打印输出 (完全复刻您的格式)
# ==============================================================================

print("\n" + "="*80)
print("A股板块轮动分析结果")
print("="*80)

# 1. 打印相关性矩阵
print("\n[表1] 板块相关性矩阵")
print("-" * 60)
# 您的数据中是 1.00, 0.78... 这里的 float_format 已设为 0.2f
print(corr_matrix)

# 2. 打印季度收益率
print("\n[表2] 季度收益率概览 (行=板块, 列=季度)")
print("-" * 120)
# 将小数转换为百分比字符串显示
q_ret_display = q_ret_table.applymap(lambda x: f"{x*100:.1f}%")
# 为了防止列太多换行，我们只打印最近8个季度 (类似您的示例 2022Q1-2023Q4)
# 如果您想看全部，去掉 .iloc[:, -8:] 即可
print(q_ret_display.iloc[:, -8:]) 

# 3. 打印风险指标
print("\n[表3] 行业板块风险指标")
print("-" * 60)
# 按照您的列顺序排序
risk_table = risk_table[['年化波动率 (%)', '最大回撤 (%)', '夏普比率']]
print(risk_table)

# 4. 打印收益指标
print("\n[表4] 行业板块收益指标")
print("-" * 80)
return_table = return_table[['累计收益率 (%)', '年化收益率 (%)', '季度平均收益率 (%)']]
print(return_table)

print("\n" + "="*80)
print("分析完成。")