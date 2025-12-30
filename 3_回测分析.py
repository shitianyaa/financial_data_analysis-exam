# -*- coding: utf-8 -*-

"""
A股行业板块轮动系统 (多因子增强版)
核心策略：多因子选股 (动量/技术/估值) + 市场状态识别 (择时)
"""

import tushare as ts
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 过滤 Pandas 的一些警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 基础配置与鉴权
# ==============================================================================

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
    print("!!! 错误：请在 .env 文件中配置 TUSHARE_TOKEN")
    sys.exit(1)

ts.set_token(MY_TOKEN)
pro = ts.pro_api()

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid', {'font.sans-serif': ['SimHei', 'Microsoft YaHei']})

# ==============================================================================
# 2. 数据获取 (带缓存)
# ==============================================================================

DATA_DIR = 'data'  # 数据存放目录

def get_data_smart(name, code, start_date, end_date, asset_type='sector'):
    """智能获取数据：优先读取本地 CSV"""
    file_path = os.path.join(DATA_DIR, f"cache_{name}.csv")
    
    # 1. 尝试读取缓存
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # print(f"  [本地] {name}")
            return df['close']
        except:
            pass

    # 2. 联网下载
    print(f"  [联网] 下载 {name} ({code})...")
    try:
        if asset_type == 'sector':
            df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date)
        else:
            df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
            
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()
            df.to_csv(file_path)
            return df['close']
    except Exception as e:
        print(f"  X {name} 下载失败: {e}")
    
    return pd.Series(dtype=float)

# --- 准备数据 ---
START_DATE = '20220101'
END_DATE = '20251226'

SECTORS = {
    '新能源': '801730.SI', '半导体': '801080.SI',
    '医药生物': '801150.SI', '食品饮料': '801120.SI',
    '有色金属': '801050.SI', '银行':     '801780.SI'
}

print(f"\n>>> 1. 正在加载数据 ({START_DATE} ~ {END_DATE})...")

# 加载板块
sector_data = {}
for name, code in SECTORS.items():
    s = get_data_smart(name, code, START_DATE, END_DATE, 'sector')
    if not s.empty: sector_data[name] = s
sector_df = pd.DataFrame(sector_data).ffill().dropna()

# 加载基准 (沪深300)
bench_series = get_data_smart('沪深300', '000300.SH', START_DATE, END_DATE, 'index')
bench_df = pd.DataFrame({'沪深300': bench_series}).ffill().dropna()

# 对齐索引
common = sector_df.index.intersection(bench_df.index)
sector_df = sector_df.loc[common]
bench_df = bench_df.loc[common]

# ==============================================================================
# 3. 多因子策略逻辑 (User Provided Logic)
# ==============================================================================

class MultiFactorRotationStrategy:
    def __init__(self, sector_data, benchmark_data):
        self.sector_df = sector_data
        self.bench_df = benchmark_data
        self.pct_change = sector_data.pct_change().fillna(0)
        self.bench_ret = benchmark_data['沪深300'].pct_change().fillna(0)
        
        # 参数配置
        self.momentum_windows = [20, 60]  # 简化窗口以适应较短回测区间
        self.rebalance_freq = 20          # 月度调仓
        
        # 初始状态
        self.current_capital = 1.0
        self.strategy_curve = [1.0]
        self.bench_curve = [1.0]
        self.target_sectors = []
        self.last_rebalance = None
        self.holdings_history = []

    def _calc_technical_score(self, prices):
        """计算技术面得分 (RSI + 均线趋势)"""
        scores = {}
        for col in prices.columns:
            p = prices[col]
            if len(p) < 20: 
                scores[col] = 0.5; continue
            
            # 简易 RSI
            delta = p.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            if loss == 0: rsi = 100
            else: rsi = 100 - (100 / (1 + gain/loss))
            
            # 均线趋势
            ma5 = p.rolling(5).mean().iloc[-1]
            ma20 = p.rolling(20).mean().iloc[-1]
            
            # 评分逻辑
            rsi_score = 1 - abs(rsi - 50)/50  # 越接近50分越高(震荡)，极端值扣分? 
            # 或者：动量策略通常喜欢强势 RSI (>50)
            rsi_score = 1 if rsi > 50 else 0
            trend_score = 1 if ma5 > ma20 else 0
            
            scores[col] = rsi_score * 0.4 + trend_score * 0.6
        return scores

    def _calc_valuation_score(self, prices):
        """计算估值得分 (相对历史位置)"""
        scores = {}
        for col in prices.columns:
            p = prices[col]
            if len(p) < 60:
                scores[col] = 0.5; continue
            
            curr = p.iloc[-1]
            high = p.rolling(60).max().iloc[-1]
            low = p.rolling(60).min().iloc[-1]
            
            # 价格越低，估值分越高
            if high != low:
                score = 1 - (curr - low) / (high - low)
            else:
                score = 0.5
            scores[col] = score
        return scores

    def _analyze_market_regime(self, idx):
        """判断市场状态 (Bull/Bear/Neutral)"""
        if idx < 60: return 'neutral'
        
        bench = self.bench_df['沪深300'].iloc[:idx+1]
        ma20 = bench.rolling(20).mean().iloc[-1]
        ma60 = bench.rolling(60).mean().iloc[-1]
        
        if ma20 > ma60: return 'bull'
        elif ma20 < ma60: return 'bear'
        return 'neutral'

    def run(self):
        print(f"\n>>> 2. 执行多因子回测 (动量+技术+估值)...")
        dates = self.sector_df.index
        
        for i in range(1, len(dates)):
            today = dates[i]
            
            # 1. 每日更新净值
            daily_ret = 0.0
            if self.target_sectors:
                # 等权重持仓
                w = 1.0 / len(self.target_sectors)
                for sec in self.target_sectors:
                    daily_ret += self.pct_change.loc[today, sec] * w
            
            self.current_capital *= (1 + daily_ret)
            self.strategy_curve.append(self.current_capital)
            
            # 基准净值
            b_ret = self.bench_ret.loc[today]
            self.bench_curve.append(self.bench_curve[-1] * (1 + b_ret))
            
            # 2. 调仓逻辑
            # 检查是否到调仓日
            is_rebalance_day = False
            if self.last_rebalance is None:
                is_rebalance_day = True
            elif (today - self.last_rebalance).days >= self.rebalance_freq:
                is_rebalance_day = True
                
            if is_rebalance_day and i > 60:
                # A. 市场状态识别
                regime = self._analyze_market_regime(i)
                
                # B. 若熊市，空仓或防御
                if regime == 'bear':
                    # 策略选择：空仓避险 (也可选择切入银行等防御板块)
                    # 这里演示：空仓
                    self.target_sectors = [] 
                    self.last_rebalance = today
                    self.holdings_history.append({'date': today, 'sectors': [], 'regime': regime})
                    continue
                
                # C. 多因子打分
                hist_price = self.sector_df.iloc[:i+1]
                
                # 因子1: 动量
                mom_score = pd.Series(0.0, index=self.sector_df.columns)
                for w in self.momentum_windows:
                    ret = (hist_price.iloc[-1] - hist_price.iloc[-w]) / hist_price.iloc[-w]
                    mom_score += ret
                
                # 因子2: 技术
                tech_score = pd.Series(self._calc_technical_score(hist_price))
                
                # 因子3: 估值
                val_score = pd.Series(self._calc_valuation_score(hist_price))
                
                # 综合打分 (牛市重动量，震荡重估值)
                if regime == 'bull':
                    final_score = mom_score * 0.5 + tech_score * 0.3 + val_score * 0.2
                else:
                    final_score = mom_score * 0.3 + tech_score * 0.3 + val_score * 0.4
                
                # D. 选股 (取Top 2)
                top_sectors = final_score.nlargest(2).index.tolist()
                
                # 只要分数大于0才买
                valid_sectors = [s for s in top_sectors if final_score[s] > 0]
                
                self.target_sectors = valid_sectors
                self.last_rebalance = today
                self.holdings_history.append({'date': today, 'sectors': valid_sectors, 'regime': regime})

        return pd.DataFrame({
            '策略净值': self.strategy_curve,
            '沪深300': self.bench_curve
        }, index=dates)

# ==============================================================================
# 4. 运行回测
# ==============================================================================

strategy = MultiFactorRotationStrategy(sector_df, bench_df)
result_df = strategy.run()

# ==============================================================================
# 5. 绩效指标计算与打印
# ==============================================================================

def calc_metrics(series):
    total_ret = series.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(series)) - 1
    daily_ret = series.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    
    # 最大回撤
    cum_max = series.cummax()
    dd = (series - cum_max) / cum_max
    max_dd = dd.min()
    
    sharpe = (ann_ret - 0.025) / ann_vol
    win_rate = (daily_ret > 0).sum() / len(daily_ret)
    
    return {
        '累计收益率': total_ret, 
        '年化收益率': ann_ret, 
        '年化波动率': ann_vol, 
        '最大回撤': max_dd, 
        '夏普比率': sharpe, 
        '胜率': win_rate
    }

m_strat = calc_metrics(result_df['策略净值'])
m_bench = calc_metrics(result_df['沪深300'])

# 生成对比表
print("\n>>> 3. 生成绩效报告...")
cols = ['累计收益率', '年化收益率', '年化波动率', '最大回撤', '夏普比率', '胜率']
rows = []

for k in cols:
    v1 = m_strat[k]
    v2 = m_bench[k]
    diff = v1 - v2
    
    # 格式化
    if k == '夏普比率':
        f1, f2, f3 = f"{v1:.2f}", f"{v2:.2f}", f"{diff:+.2f}"
    else:
        f1, f2, f3 = f"{v1:.1%}", f"{v2:.1%}", f"{diff:+.1%}"
        
    rows.append({'指标': k, '多因子策略': f1, '沪深 300': f2, '超额收益': f3})

report_df = pd.DataFrame(rows).set_index('指标')

print("\n" + "="*80)
print("             历史回测结果 (2022.01.01 - 2025.12.26)")
print("="*80)
print(report_df)
print("="*80)

# ==============================================================================
# 6. 可视化
# ==============================================================================

print("\n正在绘制分析图表...")
plt.figure(figsize=(12, 10))

# 1. 净值曲线
plt.subplot(2, 1, 1)
plt.plot(result_df.index, result_df['策略净值'], color='#d62728', lw=2, label='多因子策略')
plt.plot(result_df.index, result_df['沪深300'], color='gray', lw=1.5, alpha=0.7, label='沪深300')
plt.title('多因子策略 vs 沪深300 净值走势', fontsize=14, fontweight='bold')
plt.ylabel('净值 (Base=1.0)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# 2. 超额收益与回撤
plt.subplot(2, 1, 2)
excess = result_df['策略净值'] - result_df['沪深300']
roll_max = result_df['策略净值'].expanding().max()
dd = (result_df['策略净值'] - roll_max) / roll_max

plt.fill_between(result_df.index, dd, 0, color='green', alpha=0.3, label='策略回撤')
plt.plot(result_df.index, excess, color='blue', alpha=0.8, lw=1, label='超额净值')
plt.axhline(0, c='k', ls='--', lw=0.5)
plt.title('超额收益与回撤分析', fontsize=14, fontweight='bold')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/6_高级回测分析.png', dpi=300)
print("图表已保存: images/6_高级回测分析.png")

# 打印最近几次调仓记录
print("\n[最近调仓记录]")
recents = strategy.holdings_history[-5:]
for r in recents:
    d = r['date'].strftime('%Y-%m-%d')
    secs = ",".join(r['sectors']) if r['sectors'] else "空仓 (避险)"
    print(f"{d} | 状态: {r['regime']:<7} | 持仓: {secs}")