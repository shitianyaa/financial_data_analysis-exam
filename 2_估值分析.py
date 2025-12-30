# -*- coding: utf-8 -*-

"""
A股行业板块估值维度分析 (高级版)
对应文档章节：5.4.1 估值体系 | 5.4.2 估值水平分析
功能：获取申万行业 PE, PB，并计算历史分位数
接口：sw_daily (需 Tushare 5000 积分)
新增功能：数据缓存、用户交互选择
"""

import tushare as ts
import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
import matplotlib.pyplot as plt
import datetime

# ==============================================================================
# 1. 初始化
# ==============================================================================

def load_token_from_env():
    """从.env文件加载Tushare Token"""
    env_file = '.env'
    if not os.path.exists(env_file):
        print(f"!!! 错误：找不到 {env_file} 文件")
        print(f"请创建 {env_file} 文件并添加 TUSHARE_TOKEN=your_token_here")
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

try:
    ts.set_token(MY_TOKEN)
    pro = ts.pro_api()
    print(">>> 接口初始化成功 (高级权限模式)")
except Exception as e:
    print(f"!!! 初始化失败: {e}")
    sys.exit(1)

# ==============================================================================
# 2. 数据缓存与用户交互
# ==============================================================================

CACHE_FILE = 'data/valuation_data_cache.csv'

def load_cached_data():
    """加载缓存数据"""
    if os.path.exists(CACHE_FILE):
        try:
            df_cache = pd.read_csv(CACHE_FILE)
            # 检查缓存数据是否过期（超过7天）
            cache_date = pd.to_datetime(df_cache['cache_timestamp'].iloc[0])
            if (datetime.datetime.now() - cache_date).days < 7:
                print(f"  √ 加载缓存数据 (缓存时间: {cache_date.strftime('%Y-%m-%d %H:%M')})")
                return df_cache
            else:
                print(f"  ! 缓存数据已过期 ({cache_date.strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  ! 缓存数据加载失败: {e}")
    return None

def save_data_to_cache(df_data):
    """保存数据到缓存"""
    try:
        # 添加缓存时间戳
        df_data['cache_timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df_data.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
        print(f"  √ 数据已缓存到 {CACHE_FILE}")
    except Exception as e:
        print(f"  ! 数据缓存失败: {e}")

def ask_user_choice():
    """询问用户选择"""
    print("\n=== 数据获取选项 ===")
    print("1. 重新获取最新数据 (推荐)")
    print("2. 使用缓存数据 (如果存在)")
    
    while True:
        try:
            choice = input("请选择 (1/2, 默认1): ").strip()
            if choice == '' or choice == '1':
                return 'fresh'
            elif choice == '2':
                return 'cache'
            else:
                print("请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n程序已终止")
            sys.exit(0)

# ==============================================================================
# 3. 估值数据获取与计算 (使用 sw_daily)
# ==============================================================================

def get_valuation_analysis(use_cache=False):
    # 申万一级/二级行业代码 (SW2021版)
    # 5000积分用户可以直接获取这些代码的 PE/PB
    sectors = {
        '新能源':   '801730.SI', # 电力设备 (新能源核心板块)
        '半导体':   '801080.SI', # 电子 (半导体核心板块)
        '医药生物': '801150.SI',
        '食品饮料': '801120.SI',
        '银行':     '801780.SI',
        '有色金属': '801050.SI'
    }
    
    # 设置历史区间：过去5年
    start_date = '20200101'
    end_date = '20251226'
    
    results = []
    
    print(f"\n>>> 正在获取申万行业估值数据 ({start_date} ~ {end_date})...")
    print("    (调用 sw_daily 接口)")
    
    for name, code in sectors.items():
        try:
            # sw_daily 包含 fields: open, close, pe, pb, vol, amount
            # 注意：sw_daily 单次限量 4000 行，我们按单只代码取5年数据(~1200行)是安全的
            df = pro.sw_daily(
                ts_code=code, 
                start_date=start_date, 
                end_date=end_date, 
                fields='trade_date,pe,pb,close'
            )
            
            if df.empty:
                print(f"  X {name} ({code}): 无数据 (请确认积分是否生效)")
                continue
            
            # 确保按日期排序
            df = df.sort_values('trade_date')
            
            # 清洗数据：去除 PE/PB 为 0 或空的异常值（有些历史时期可能缺失）
            df_valid = df.replace(0, np.nan)
            
            # 获取最新一天的估值数据
            current = df.iloc[-1]
            cur_pe = current['pe']
            cur_pb = current['pb']
            
            # --- 核心算法：计算历史分位数 ---
            # percentilerank logic
            if pd.notna(cur_pe):
                pe_hist = df_valid['pe'].dropna()
                if len(pe_hist) > 0:
                    pe_percentile = stats.percentileofscore(pe_hist, cur_pe)
                else:
                    pe_percentile = 0
            else:
                pe_percentile = 0
                
            if pd.notna(cur_pb):
                pb_hist = df_valid['pb'].dropna()
                if len(pb_hist) > 0:
                    pb_percentile = stats.percentileofscore(pb_hist, cur_pb)
                else:
                    pb_percentile = 0
            
            # 估算 PEG (PE / Growth)
            estimated_growth = {
                '新能源': 25, '半导体': 25, '医药生物': 20, 
                '食品饮料': 15, '银行': 5, '有色金属': 12
            }
            g = estimated_growth.get(name, 10)
            peg = cur_pe / g if (g > 0 and pd.notna(cur_pe)) else np.nan
            
            results.append({
                '板块': name,
                '代码': code,
                'PE': round(cur_pe, 2) if pd.notna(cur_pe) else None,
                'PE历史分位': f"{pe_percentile:.1f}%",
                'PB': round(cur_pb, 2) if pd.notna(cur_pb) else None,
                'PB历史分位': f"{pb_percentile:.1f}%",
                'PEG (预估)': round(peg, 2) if pd.notna(peg) else "-"
            })
            
            print(f"  √ {name}: PE={cur_pe:.1f}, 分位={pe_percentile:.1f}%")
            
        except Exception as e:
            print(f"  ! {name} 获取失败: {e}")
            
    return pd.DataFrame(results)

# ==============================================================================
# 4. 生成表格与图表
# ==============================================================================

def generate_report(df_valuation):
    """生成估值分析报告"""
    if not df_valuation.empty:
        # 排序
        target_order = ['新能源', '半导体', '医药生物', '食品饮料', '银行', '有色金属']
        df_valuation['sort_cat'] = pd.Categorical(df_valuation['板块'], categories=target_order, ordered=True)
        df_valuation = df_valuation.sort_values('sort_cat').drop('sort_cat', axis=1)

        print("\n" + "="*85)
        print("             5.4.2 申万行业板块估值分析 (Source: sw_daily)")
        print("="*85)
        print(df_valuation.to_string(index=False))
        print("="*85)

        # 绘图
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax1 = plt.subplots(figsize=(12, 6))

        x = df_valuation['板块']
        y1 = df_valuation['PE']
        y2 = df_valuation['PE历史分位'].str.rstrip('%').astype(float)

        # 柱状图：PE
        bars = ax1.bar(x, y1, color='#5DADE2', alpha=0.8, label='PE (当前)')
        ax1.set_ylabel('PE (倍)', fontsize=12)
        ax1.set_title('申万一级行业当前PE与历史分位对比 (sw_daily接口)', fontsize=14)

        # 数值标签
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # 折线图：分位
        ax2 = ax1.twinx()
        ax2.plot(x, y2, color='#E74C3C', marker='D', lw=2, ms=6, label='历史分位(%)')
        ax2.set_ylabel('历史分位 (%)', fontsize=12)
        ax2.set_ylim(0, 110)
        
        for i, val in enumerate(y2):
            ax2.text(i, val + 3, f'{val:.0f}%', ha='center', color='#E74C3C', fontsize=9, fontweight='bold')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig('images/5_板块估值分析_SW.png', dpi=300)
        print("\n>>> 分析图表已保存为 images/5_板块估值分析_SW.png")
    else:
        print("\n!!! 未获取到数据。尽管您有5000积分，请确认：")
        print("1. Token 是否正确配置在 .env 文件中")
        print("2. 您的积分是否已到账 (可登录 tushare.pro 个人主页查看)")

if __name__ == "__main__":
    # 检查缓存数据
    cached_data = load_cached_data()
    
    # 询问用户选择
    user_choice = ask_user_choice()
    
    if user_choice == 'cache' and cached_data is not None:
        print("\n>>> 使用缓存数据生成报告...")
        # 移除缓存时间戳列
        df_valuation = cached_data.drop('cache_timestamp', axis=1, errors='ignore')
    else:
        print("\n>>> 重新获取最新数据...")
        df_valuation = get_valuation_analysis()
        
        # 如果获取到新数据，保存到缓存
        if not df_valuation.empty:
            save_data_to_cache(df_valuation)
    
    # 生成报告
    generate_report(df_valuation)