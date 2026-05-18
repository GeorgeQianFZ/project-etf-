"""
子任务 2a：ETF 申赎套利策略（511130 ETF，2026-01-01 ~ 2026-02-28）

策略逻辑：
  当 ETF 二级市场价格偏离 IOPV（基金份额参考净值）超过交易成本时触发套利。
  - 溢价：ETF_close - IOPV > TC → 买债申购ETF → 卖ETF
  - 折价：IOPV - ETF_close > TC → 买ETF赎回 → 卖债

关键细节：
  1. IOPV = (Σ 债券全价_i × PCF数量_i × 1000 + 现金差额) / 份额篮大小(10000)
  2. 债券全价 = 净价 + 应计利息；行情数据为净价（陷阱）
  3. 应计利息按 A/365F 日计数惯例计算
  4. 现金差额取前一交易日公告值（申赎套利用 T-1 日的 PCF / 现金差额）
  5. 每笔套利固定为1篮子，每日最多执行一次，模拟 T+1 申赎流程
  6. 交易成本：ETF 万0.5 双边（实际单边买或卖），债券万分之0.05
"""

from datetime import datetime, timedelta
from pathlib import Path

import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/System/Library/Fonts/STHeiti Light.ttc")
matplotlib.rcParams["font.family"] = "Heiti TC"
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("/Users/bulusiweien/Downloads/钱方舟测试结果")
BASE_DIR = Path("/Users/bulusiweien/Downloads/汇远盈测试题需要的数据")

# ── 债券基本条款 ───────────────────────────────────────────────────────────────
# 付息频率均为半年，A/365F日计数
BOND_INFO = {
    "019742.SH": {
        "coupon_rate": 0.0257,
        "coupon_dates_md": [(5, 20), (11, 20)],   # (月, 日)
        "issue_date": pd.Timestamp("2024-05-20"),
    },
    "019776.SH": {
        "coupon_rate": 0.0188,
        "coupon_dates_md": [(4, 25), (10, 25)],
        "issue_date": pd.Timestamp("2025-04-25"),
    },
    "019789.SH": {
        "coupon_rate": 0.0215,
        "coupon_dates_md": [(2, 25), (8, 25)],
        "issue_date": pd.Timestamp("2025-08-25"),
    },
}

PCF_PATH = BASE_DIR / "其他文件" / "511130_PCF清单_2024-12-31_2026-03-20.csv"
CASH_PATH = BASE_DIR / "其他文件" / "511130_现金差额_2024-12-31_2026-03-20.csv"


# ── 应计利息计算 ───────────────────────────────────────────────────────────────

def calc_accrued_interest(bond_code: str, trade_date: pd.Timestamp) -> float:
    """
    计算指定债券在 trade_date 的应计利息（每100元面值）。
    日计数基准：A/365F（实际天数 / 365，分母固定为365）。
    """
    info = BOND_INFO[bond_code]
    coupon_rate = info["coupon_rate"]
    issue_date = info["issue_date"]

    # 找出 trade_date 前最近的付息日（或起息日）
    last_coupon = issue_date
    for year in range(trade_date.year - 1, trade_date.year + 1):
        for month, day in info["coupon_dates_md"]:
            try:
                cdt = pd.Timestamp(year, month, day)
                if issue_date <= cdt <= trade_date:
                    if cdt > last_coupon:
                        last_coupon = cdt
            except ValueError:
                continue

    days = (trade_date - last_coupon).days
    # A/365F：每期票息 = 年票面利率 / 付息频率 = coupon_rate / 2 * 100
    coupon_per_period = coupon_rate * 100 / 2          # 每期票息 per 100
    days_per_period = 365 / 2                           # 每期天数（固定365）
    accrued = coupon_per_period * days / days_per_period
    return accrued


# ── PCF / 现金差额加载 ─────────────────────────────────────────────────────────

def load_pcf() -> pd.DataFrame:
    """加载PCF成分数据，标准化债券代码格式（XSHG→SH）。"""
    df = pd.read_csv(PCF_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # 统一代码格式：019742.XSHG → 019742.SH
    df["bond_code"] = df["stock_code"].str.replace(".XSHG", ".SH", regex=False)
    return df[["date", "bond_code", "stock_amount"]]


def load_cash_diff() -> pd.DataFrame:
    """加载现金差额数据。pre_date 即前一交易日，对应当日申赎。"""
    df = pd.read_csv(CASH_PATH)
    df["pre_date"] = pd.to_datetime(df["pre_date"])
    return df[["pre_date", "cash_component", "unit_subscribe_redeem", "nav_per_basket"]]


def get_pcf_for_date(pcf_df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.Series:
    """取 trade_date 前最近一个公告日的PCF（PCF当日公告，次日生效）。"""
    prev = pcf_df[pcf_df["date"] < trade_date]
    if prev.empty:
        prev = pcf_df[pcf_df["date"] <= trade_date]
    latest = prev["date"].max()
    return pcf_df[pcf_df["date"] == latest]


def get_cash_for_date(cash_df: pd.DataFrame, trade_date: pd.Timestamp) -> float:
    """取 trade_date 对应的现金差额（以 pre_date < trade_date 的最新一条）。"""
    prev = cash_df[cash_df["pre_date"] < trade_date]
    if prev.empty:
        prev = cash_df[cash_df["pre_date"] <= trade_date]
    if prev.empty:
        return 0.0
    row = prev.sort_values("pre_date").iloc[-1]
    return float(row["cash_component"])


def get_unit_for_date(cash_df: pd.DataFrame, trade_date: pd.Timestamp) -> int:
    """取 unit_subscribe_redeem（篮子份额）。"""
    prev = cash_df[cash_df["pre_date"] < trade_date]
    if prev.empty:
        prev = cash_df[cash_df["pre_date"] <= trade_date]
    if prev.empty:
        return 10000
    return int(prev.sort_values("pre_date").iloc[-1]["unit_subscribe_redeem"])


# ── IOPV 序列预计算 ────────────────────────────────────────────────────────────

def build_iopv_series(start_date: str = "2026-01-01",
                      end_date: str = "2026-02-28") -> pd.DataFrame:
    """
    构造以1分钟为粒度的 IOPV 序列，供回测使用。

    IOPV per share = (Σ 全价_i × 数量_i × 1000 + 现金差额) / 篮子份额

    Returns
    -------
    DataFrame，index=trade_time，列：etf_close, iopv, premium
    """
    print("\n[2a] 构建 IOPV 序列...")

    # --- ETF 数据（1分钟重采样）---
    etf_raw = pd.read_csv(
        OUTPUT_DIR / "511130_20260101_20260228_merged.CSV",
        parse_dates=["trade_time"],
    )
    etf_raw = etf_raw.set_index("trade_time").sort_index()
    etf_1min = etf_raw["close"].resample("1min").last().dropna()
    etf_1min.name = "etf_close"

    # --- 债券数据（1分钟重采样，前向填充）---
    bond_codes = ["019742.SH", "019776.SH", "019789.SH"]
    bond_1min = {}
    for code in bond_codes:
        code_str = code.split(".")[0]
        df = pd.read_csv(
            OUTPUT_DIR / f"{code_str}_20260101_20260228_merged.CSV",
            parse_dates=["trade_time"],
        )
        df = df.set_index("trade_time").sort_index()
        s = df["close"].resample("1min").last().ffill()
        bond_1min[code] = s

    # --- PCF / 现金差额 ---
    pcf_df = load_pcf()
    cash_df = load_cash_diff()

    # --- 逐分钟计算 IOPV ---
    records = []
    for ts, etf_price in etf_1min.items():
        if pd.isna(etf_price):
            continue
        trade_date = pd.Timestamp(ts).normalize()

        # 获取当日 PCF 成分
        pcf_today = get_pcf_for_date(pcf_df, trade_date)
        cash = get_cash_for_date(cash_df, trade_date)
        unit = get_unit_for_date(cash_df, trade_date)

        basket_value = 0.0
        for _, row in pcf_today.iterrows():
            bcode = row["bond_code"]
            qty = float(row["stock_amount"])  # 张（1张=1000元面值）

            # 当前净价（1分钟重采样后的最新值）
            if bcode in bond_1min and ts in bond_1min[bcode].index:
                clean_px = bond_1min[bcode].get(ts, np.nan)
            else:
                # 若债券在该时刻无报价，用最近值
                bseries = bond_1min.get(bcode)
                if bseries is None:
                    continue
                avail = bseries[:ts].dropna()
                clean_px = avail.iloc[-1] if not avail.empty else np.nan

            if np.isnan(clean_px):
                continue

            # 应计利息（per 100元面值）
            ai = calc_accrued_interest(bcode, trade_date)

            # 全价 = 净价 + 应计利息
            full_px = clean_px + ai

            # 该债券在篮子中的价值（元）= 全价/100 × 数量 × 1000
            basket_value += (full_px / 100) * qty * 1000

        # 现金差额
        basket_value += cash

        # IOPV per share
        iopv = basket_value / unit if unit > 0 else np.nan

        records.append({
            "trade_time": ts,
            "etf_close": etf_price,
            "iopv": iopv,
        })

    df_out = pd.DataFrame(records).set_index("trade_time")
    df_out = df_out.dropna()
    df_out["premium"] = df_out["etf_close"] - df_out["iopv"]

    print(f"  IOPV序列行数：{len(df_out)}")
    print(f"  ETF均价：{df_out['etf_close'].mean():.4f}")
    print(f"  IOPV均值：{df_out['iopv'].mean():.4f}")
    print(f"  溢价均值：{df_out['premium'].mean():.6f}")
    print(f"  溢价标准差：{df_out['premium'].std():.6f}")

    df_out.to_csv(OUTPUT_DIR / "511130_iopv_series.csv")
    return df_out


# ── Backtrader 数据馈送 ───────────────────────────────────────────────────────

class ETFArbitrageData(bt.feeds.PandasData):
    """自定义数据馈送：close=ETF价格，openinterest 复用为 IOPV。"""
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "etf_close"),
        ("volume", -1),
        ("openinterest", "iopv"),
    )


# ── 套利策略类 ────────────────────────────────────────────────────────────────

class ETFArbitrageStrategy(bt.Strategy):
    """
    ETF 申赎套利策略。

    Parameters
    ----------
    max_capital : float
        策略资金上限（元）
    max_baskets_per_trade : int
        单次最大套利篮子数
    etf_fee_rate : float
        ETF 单边费率（万0.5 = 0.00005）
    bond_fee_rate : float
        债券单边费率（万分之0.05 = 0.000005）
    cooldown_bars : int
        两次交易之间最少间隔 bar 数（防止同一价差区间重复套利）
    """
    params = (
        ("max_capital", 10_000_000),
        ("max_baskets_per_trade", 5),
        ("etf_fee_rate", 0.5 / 10000),
        ("bond_fee_rate", 0.05 / 10000),
        ("cooldown_bars", 60),       # 60分钟冷却
    )

    def __init__(self):
        self.iopv_line = self.data.openinterest
        self.trades_log = []
        self.equity_curve = []
        self.total_pnl = 0.0
        self.trade_count = 0
        self._bars_since_last_trade = self.p.cooldown_bars  # 初始允许立即交易
        self._last_trade_date = None

    def next(self):
        etf = self.data.close[0]
        iopv = self.iopv_line[0]
        self._bars_since_last_trade += 1

        if etf <= 0 or iopv <= 0 or np.isnan(iopv):
            self._record_equity(etf, iopv)
            return

        # 可投入的最大篮子数（资金限制）
        basket_etf_cost = etf * 10000     # 按ETF价值估算1篮子成本
        max_by_capital = int(self.p.max_capital / basket_etf_cost) if basket_etf_cost > 0 else 0
        n = min(self.p.max_baskets_per_trade, max_by_capital)
        if n == 0:
            self._record_equity(etf, iopv)
            return

        # 每篮子交易成本（ETF + 债券各单边）
        unit = 10000  # 篮子份额
        tc_etf = etf * unit * self.p.etf_fee_rate      # 卖出 or 买入 ETF
        tc_bond = iopv * unit * self.p.bond_fee_rate   # 买入 or 卖出 债券
        tc_per_basket = tc_etf + tc_bond

        premium = etf - iopv
        tc_per_share = tc_per_basket / unit

        # 每日最多一次套利（同一日期不重复）
        cur_date = self.data.datetime.date(0)
        if cur_date == self._last_trade_date:
            self._record_equity(etf, iopv)
            return

        if premium > tc_per_share and self._bars_since_last_trade >= self.p.cooldown_bars:
            # ETF 溢价套利：买债申购 → 卖ETF
            pnl = (premium - tc_per_share) * unit * n
            self._execute_trade("premium", etf, iopv, n, pnl, tc_per_basket * n)

        elif -premium > tc_per_share and self._bars_since_last_trade >= self.p.cooldown_bars:
            # ETF 折价套利：买ETF赎回 → 卖债
            pnl = (-premium - tc_per_share) * unit * n
            self._execute_trade("discount", etf, iopv, n, pnl, tc_per_basket * n)

        self._record_equity(etf, iopv)

    def _execute_trade(self, trade_type, etf, iopv, n_baskets, pnl, tc_total):
        self.total_pnl += pnl
        self.trade_count += 1
        self._bars_since_last_trade = 0
        self._last_trade_date = self.data.datetime.date(0)
        self.trades_log.append({
            "datetime": self.data.datetime.datetime(0),
            "type": trade_type,
            "etf_price": etf,
            "iopv": iopv,
            "premium_bps": (etf - iopv) / iopv * 10000,
            "n_baskets": n_baskets,
            "pnl": pnl,
            "tc": tc_total,
            "cum_pnl": self.total_pnl,
        })

    def _record_equity(self, etf, iopv):
        self.equity_curve.append({
            "datetime": self.data.datetime.datetime(0),
            "etf": etf,
            "iopv": iopv,
            "premium": etf - iopv,
            "cum_pnl": self.total_pnl,
        })


# ── 绩效指标计算 ───────────────────────────────────────────────────────────────

def calc_performance(equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                     initial_capital: float = 10_000_000) -> dict:
    """计算回测绩效指标。"""
    eq = equity_df.set_index("datetime")["cum_pnl"]

    # 总收益
    total_pnl = eq.iloc[-1]
    abs_return = total_pnl

    # 年化收益（实际交易天数 → 年化）
    days = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days
    annual_return = (total_pnl / initial_capital) / max(days, 1) * 252

    # 日收益序列（按日最后一个 cum_pnl 差分）
    daily_pnl = eq.resample("D").last().diff().dropna()
    daily_ret = daily_pnl / initial_capital

    # 夏普比率（年化，无风险利率=0）
    if daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # 最大回撤
    cum_value = initial_capital + eq
    roll_max = cum_value.expanding().max()
    drawdown = (cum_value - roll_max) / roll_max
    max_dd = drawdown.min()

    return {
        "绝对收益(元)": round(abs_return, 2),
        "年化收益率": round(annual_return * 100, 4),
        "夏普比率": round(sharpe, 4),
        "最大回撤": round(max_dd * 100, 4),
        "交易次数": len(trades_df),
    }


# ── 主回测入口 ────────────────────────────────────────────────────────────────

def run_etf_arbitrage_backtest(
    max_capital: float = 10_000_000,
    max_baskets_per_trade: int = 5,
) -> dict:
    """
    运行 511130 ETF 申赎套利回测（2026-01-01 ~ 2026-02-28）。

    Returns
    -------
    dict，包含绩效指标 + 权益曲线 DataFrame + 交易记录 DataFrame
    """
    print("\n" + "=" * 60)
    print("子任务 2a：ETF 申赎套利回测（511130）")
    print("=" * 60)

    # 1. 构建 IOPV 序列
    iopv_df = build_iopv_series()

    # 2. 构造 backtrader 数据馈送
    feed_df = iopv_df[["etf_close", "iopv"]].copy()
    feed_df["open"] = feed_df["etf_close"]
    feed_df["high"] = feed_df["etf_close"]
    feed_df["low"] = feed_df["etf_close"]
    feed_df.index.name = "datetime"

    data_feed = ETFArbitrageData(dataname=feed_df)

    # 3. 配置 Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(
        ETFArbitrageStrategy,
        max_capital=max_capital,
        max_baskets_per_trade=max_baskets_per_trade,
    )
    cerebro.broker.setcash(max_capital)

    # 4. 运行回测
    results = cerebro.run()
    strat = results[0]

    # 5. 整理结果
    equity_df = pd.DataFrame(strat.equity_curve)
    trades_df = pd.DataFrame(strat.trades_log)

    perf = calc_performance(equity_df, trades_df, initial_capital=max_capital)

    print("\n── 绩效指标 ──")
    for k, v in perf.items():
        print(f"  {k}: {v}")

    # 6. 保存结果
    trades_df.to_csv(OUTPUT_DIR / "2a_trades.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUTPUT_DIR / "2a_equity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([perf]).to_csv(
        OUTPUT_DIR / "2a_performance.csv", index=False, encoding="utf-8-sig"
    )

    # 7. 画图
    _plot_arbitrage_results(equity_df, trades_df, iopv_df, perf)

    return {"perf": perf, "equity": equity_df, "trades": trades_df}


def _plot_arbitrage_results(equity_df, trades_df, iopv_df, perf):
    """生成套利策略结果图表。"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("511130 ETF 申赎套利策略回测结果", fontsize=14, fontweight="bold")

    # 图1：ETF 价格 vs IOPV
    ax1 = axes[0]
    idx = iopv_df.index.to_numpy()
    ax1.plot(idx, iopv_df["etf_close"].to_numpy(), label="ETF Close", color="steelblue", lw=0.8)
    ax1.plot(idx, iopv_df["iopv"].to_numpy(), label="IOPV", color="orange", lw=0.8, ls="--")
    ax1.set_title("ETF 价格 vs IOPV")
    ax1.legend(fontsize=8)
    ax1.set_ylabel("价格 (元)")
    ax1.grid(alpha=0.3)

    # 图2：溢价（bps）
    ax2 = axes[1]
    premium_bps = (iopv_df["premium"] / iopv_df["iopv"] * 10000).to_numpy()
    ax2.plot(idx, premium_bps, color="purple", lw=0.6, label="溢价 (bps)")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title("ETF 溢价/折价（基点）")
    ax2.set_ylabel("bps")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 图3：累计收益曲线
    ax3 = axes[2]
    if not equity_df.empty:
        eq_times = pd.to_datetime(equity_df["datetime"]).to_numpy()
        eq_pnl = equity_df["cum_pnl"].to_numpy()
        ax3.plot(eq_times, eq_pnl, color="green", lw=1.2, label="累计收益",)
    sharpe_str = f"Sharpe={perf['夏普比率']:.2f}"
    dd_str = f"MaxDD={perf['最大回撤']:.2f}%"
    ax3.set_title(f"累计收益曲线  |  {sharpe_str}  |  {dd_str}")
    ax3.set_ylabel("收益 (元)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "2a_arbitrage_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存：{out.name}")


if __name__ == "__main__":
    run_etf_arbitrage_backtest()
