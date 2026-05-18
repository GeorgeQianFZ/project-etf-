"""
子任务 2b：ETF-期货配对交易策略（基于 OU 过程）
标的：511090.SH ETF vs TL 国债期货主力合约
回测期：2025-07-01 ~ 2025-12-31

策略逻辑：
1. 价差序列建模
   - 将 ETF 和期货数据重采样至 1 分钟 OHLCV
   - 用训练集（前50%数据）通过 OLS 回归估算对冲比率 β
   - 价差 spread = ETF_mid - β × Futures_adj_mid
   - 对 spread 拟合 OU 过程，估算参数 θ（均值回归速度）、μ（长期均值）、σ（波动率）

2. 交易信号
   - z_score = (spread - μ) / σ
   - z_score > entry_z  → 卖空价差（做空ETF + 做多期货）
   - z_score < -entry_z → 做多价差（做多ETF + 做空期货）
   - |z_score| < exit_z → 平仓
   - 每 180 分钟动态重新估算 OU 参数

3. 仓位管理
   - 期货最大持仓：8 手
   - ETF 持仓与期货持仓按对冲比率 β 对冲
   - 仓位大小 = min(floor(|z_score| / entry_z × max_futures), max_futures) 手

陷阱处理：
- 期货换月跳价通过后复权处理（已在 data_processor.py 中完成）
"""

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
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

OUTPUT_DIR = Path("/Users/bulusiweien/Downloads/钱方舟测试结果")

# TL 期货合约乘数（每点 10000 元）
FUTURES_MULTIPLIER = 10000


# ── OU 参数估计 ────────────────────────────────────────────────────────────────

def estimate_ou_params(spread: np.ndarray, dt: float = 1.0) -> dict:
    """
    用 AR(1) 回归估算 OU 过程参数。

    离散化：X_{t+1} = a * X_t + b + ε
    还原：
      θ = -ln(a) / dt
      μ = b / (1 - a)
      σ_ε = std(残差)
      σ_eq = σ_ε / sqrt(1 - a^2)   （均衡标准差）

    Parameters
    ----------
    spread : 价差时间序列（numpy array）
    dt     : 时间步长（单位：分钟，默认1分钟）

    Returns
    -------
    dict，包含 theta, mu, sigma, sigma_eq, half_life
    """
    if len(spread) < 30:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan,
                "sigma_eq": np.nan, "half_life": np.nan}

    x = spread[:-1]
    y = spread[1:]

    # OLS 回归
    X = add_constant(x)
    try:
        model = OLS(y, X).fit()
        b, a = model.params   # intercept, slope
    except Exception:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan,
                "sigma_eq": np.nan, "half_life": np.nan}

    if a <= 0 or a >= 1:
        # OU 过程要求 0 < a < 1（稳定均值回归）
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan,
                "sigma_eq": np.nan, "half_life": np.nan}

    theta = -np.log(a) / dt
    mu = b / (1 - a)
    sigma_eps = np.std(model.resid, ddof=1)
    sigma_eq = sigma_eps / np.sqrt(1 - a ** 2)
    half_life = np.log(2) / theta if theta > 0 else np.inf  # 半衰期（分钟）

    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma_eps,
        "sigma_eq": sigma_eq,
        "half_life": half_life,
        "a": a,
        "b": b,
    }


# ── 价差序列构造 ───────────────────────────────────────────────────────────────

def build_pairs_dataset(start_date: str = "2025-07-01",
                        end_date: str = "2025-12-31") -> tuple:
    """
    将 511090 ETF 与 TL 期货（后复权）数据对齐至 1 分钟，
    构造价差序列并返回合并 DataFrame。
    """
    print("\n[2b] 构建配对数据集...")

    # --- 加载 ETF 511090（已保存至 output）---
    etf_raw = pd.read_csv(
        OUTPUT_DIR / "511090_20250701_20251231_merged.CSV",
        parse_dates=["trade_time"],
    )
    etf_raw = etf_raw.set_index("trade_time").sort_index()
    etf_1min = etf_raw["close"].resample("1min").last().dropna()
    etf_1min.name = "etf_close"

    # --- 加载 TL 期货（后复权，已保存至 output）---
    tl_raw = pd.read_csv(
        OUTPUT_DIR / "TL_futures_20250701_20251231_merged.CSV",
        parse_dates=["trade_time"],
    )
    tl_raw = tl_raw.set_index("trade_time").sort_index()
    # 使用后复权中间价
    tl_1min = tl_raw["adj_mid_price"].resample("1min").last().dropna()
    tl_1min.name = "tl_close"

    # --- 对齐合并 ---
    merged = pd.concat([etf_1min, tl_1min], axis=1).dropna()

    # 过滤交易时段（9:30-11:30, 13:00-15:00）
    t_min = merged.index.hour * 60 + merged.index.minute
    morning = (t_min >= 9 * 60 + 30) & (t_min <= 11 * 60 + 30)
    afternoon = (t_min >= 13 * 60) & (t_min <= 15 * 60)
    merged = merged[morning | afternoon].copy()

    # --- 训练集（前50%）估计对冲比率 β ---
    n_train = len(merged) // 2
    train = merged.iloc[:n_train]
    X_train = add_constant(train["tl_close"].values)
    model_beta = OLS(train["etf_close"].values, X_train).fit()
    alpha_0, beta = model_beta.params
    print(f"  OLS 对冲比率 β = {beta:.6f}，截距 α = {alpha_0:.4f}")
    print(f"  R² = {model_beta.rsquared:.4f}")

    # --- 构造价差序列 ---
    merged["spread"] = merged["etf_close"] - beta * merged["tl_close"]

    # 保存
    merged.to_csv(OUTPUT_DIR / "511090_TL_pairs_1min.csv")
    print(f"  合并数据行数：{len(merged)}，训练集：{n_train}，测试集：{len(merged)-n_train}")

    return merged, beta, n_train


# ── Backtrader 数据馈送 ───────────────────────────────────────────────────────

class PairsData(bt.feeds.PandasData):
    """
    配对数据馈送。
      close       = ETF 价格
      openinterest = 期货价格（后复权）
    额外通过 params 映射 spread 列到 'high' 字段（借用）。
    """
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "spread"),       # 借用 high 传递 spread
        ("low", "low"),
        ("close", "etf_close"),
        ("volume", -1),
        ("openinterest", "tl_close"),  # 期货价格
    )


# ── 配对交易策略 ───────────────────────────────────────────────────────────────

class OUPairsStrategy(bt.Strategy):
    """
    基于 OU 过程的 ETF-期货配对交易策略。

    Parameters
    ----------
    beta           : 对冲比率（ETF/Futures），OLS 训练集估计
    entry_z        : 入场 z-score 阈值（默认 1.5）
    exit_z         : 出场 z-score 阈值（默认 0.5）
    max_futures    : 最大期货持仓（手，默认 8）
    max_capital    : 策略最大资金（元，默认 1000万）
    update_freq    : OU 参数更新频率（bar数，默认 180 分钟 = 180 bars）
    train_bars     : 训练集长度（bar数），训练期内不交易
    ou_window      : 动态估计使用最近多少 bar 的数据
    etf_fee_rate   : ETF 单边费率
    futures_fee    : 期货单笔手续费（元/手）
    """
    params = (
        ("beta", 1.0),
        ("entry_z", 1.5),
        ("exit_z", 0.5),
        ("max_futures", 8),
        ("max_capital", 10_000_000),
        ("update_freq", 180),
        ("train_bars", 0),
        ("ou_window", 1200),          # 用最近 1200 根 bar（≈20小时）估计参数
        ("init_theta", 0.01),         # 初始 OU 参数（由训练集预估后传入）
        ("init_mu", 0.0),
        ("init_sigma_eq", 1.0),
        ("etf_fee_rate", 0.5 / 10000),
        ("futures_fee_per_lot", 3.0), # 期货每手手续费（元）
    )

    def __init__(self):
        # 当前 OU 参数（用训练集预估值初始化，避免测试期前几根 bar 无参数）
        self.ou = {
            "theta": self.p.init_theta,
            "mu": self.p.init_mu,
            "sigma_eq": self.p.init_sigma_eq,
        }
        self.bars_since_update = 0

        # 当前仓位（正=多，负=空）
        self.futures_pos = 0   # 期货手数（正=多）
        self.etf_pos_shares = 0  # ETF 份额（正=多）

        # 记录
        self.equity_curve = []
        self.trades_log = []
        self.ou_params_log = []
        self.zscore_log = []
        self.total_pnl = 0.0
        self.trade_count = 0

        # 持仓市值追踪（用于绩效计算）
        self._entry_etf_price = None
        self._entry_tl_price = None

    # ------------------------------------------------------------------
    def _update_ou_params(self, bar_idx: int):
        """从历史价差数据中重新估计 OU 参数。"""
        # 可访问的最大历史深度 = 已处理的 bar 数（backtrader 内部缓冲）
        available = min(self.p.ou_window, bar_idx)
        if available < 60:
            return

        # self.data.high[-i]：i=0 为当前，i=1 为前1根，依此类推
        spread_hist = np.array([self.data.high[-i] for i in range(available)])[::-1]

        if np.any(np.isnan(spread_hist)):
            spread_hist = spread_hist[~np.isnan(spread_hist)]

        if len(spread_hist) < 60:
            return

        ou = estimate_ou_params(spread_hist, dt=1.0)
        if not np.isnan(ou["theta"]):
            self.ou = ou
            self.ou_params_log.append({
                "datetime": self.data.datetime.datetime(0),
                "theta": ou["theta"],
                "mu": ou["mu"],
                "sigma_eq": ou["sigma_eq"],
                "half_life_min": ou["half_life"],
            })

    # ------------------------------------------------------------------
    def next(self):
        bar_idx = len(self.data) - 1

        # 训练期内只更新参数，不交易
        if bar_idx < self.p.train_bars:
            # 训练期内仅在最后一根 bar 做一次参数更新（防止重复）
            if bar_idx == self.p.train_bars - 1:
                self._update_ou_params(bar_idx)
                hl = self.ou.get("half_life", float("nan"))
                print(f"  [训练完成 bar={bar_idx}] θ={self.ou['theta']:.4f}，"
                      f"μ={self.ou['mu']:.4f}，σ_eq={self.ou['sigma_eq']:.4f}，"
                      f"半衰期={hl:.1f}分钟")
            self._log_equity()
            return

        # 动态更新 OU 参数（每 update_freq bars 更新一次）
        self.bars_since_update += 1
        if self.bars_since_update >= self.p.update_freq:
            self._update_ou_params(bar_idx)
            self.bars_since_update = 0

        etf = self.data.close[0]
        tl = self.data.openinterest[0]
        spread = self.data.high[0]

        if np.isnan(spread) or self.ou["sigma_eq"] <= 0 or np.isnan(self.ou["mu"]):
            self._log_equity()
            return

        # z-score
        z = (spread - self.ou["mu"]) / self.ou["sigma_eq"]

        self.zscore_log.append({
            "datetime": self.data.datetime.datetime(0),
            "spread": spread,
            "zscore": z,
            "mu": self.ou["mu"],
            "futures_pos": self.futures_pos,
            "etf_pos": self.etf_pos_shares,
        })

        # ── 信号生成 ──────────────────────────────────────────────────
        target_futures = self._calc_target_futures(z)

        if self.futures_pos == 0:
            # 当前空仓，检查入场
            if target_futures != 0:
                self._open_position(target_futures, z, etf, tl, spread)
        else:
            # 当前有仓位，检查出场或反转
            if abs(z) < self.p.exit_z:
                self._close_position(etf, tl, spread)
            elif target_futures != 0 and np.sign(target_futures) != np.sign(self.futures_pos):
                # 反转信号
                self._close_position(etf, tl, spread)
                self._open_position(target_futures, z, etf, tl, spread)

        self._log_equity()

    # ------------------------------------------------------------------
    def _calc_target_futures(self, z: float) -> int:
        """根据 z-score 计算目标期货手数（含方向和比例仓位）。"""
        if abs(z) < self.p.entry_z:
            return 0

        # 仓位比例随 z-score 绝对值线性增加
        size = min(
            int(abs(z) / self.p.entry_z * self.p.max_futures),
            self.p.max_futures,
        )

        # z > 0 → 价差偏高 → 卖空价差 → 做空ETF + 做多期货 → futures_pos > 0
        # z < 0 → 价差偏低 → 做多价差 → 做多ETF + 做空期货 → futures_pos < 0
        direction = 1 if z > 0 else -1
        return direction * size

    # ------------------------------------------------------------------
    def _open_position(self, target_futures: int, z: float,
                       etf: float, tl: float, spread: float):
        """开仓：期货 + ETF 对冲仓位。"""
        # 期货每手价值
        futures_lot_value = tl * FUTURES_MULTIPLIER
        # ETF 对冲份额 = futures_lots × β × futures_contract_value / etf_price
        etf_shares = int(abs(target_futures) * self.p.beta * futures_lot_value / etf)

        # 资金检查
        if etf * etf_shares > self.p.max_capital:
            etf_shares = int(self.p.max_capital / etf / 100) * 100

        self.futures_pos = target_futures
        self.etf_pos_shares = -target_futures / abs(target_futures) * etf_shares  # 方向相反
        self._entry_etf_price = etf
        self._entry_tl_price = tl

        # 计算交易成本
        tc = (etf * etf_shares * self.p.etf_fee_rate
              + abs(target_futures) * self.p.futures_fee_per_lot)

        self.trade_count += 1
        self.trades_log.append({
            "datetime": self.data.datetime.datetime(0),
            "action": "open",
            "direction": "short_spread" if target_futures > 0 else "long_spread",
            "futures_lots": target_futures,
            "etf_shares": int(self.etf_pos_shares),
            "etf_price": etf,
            "tl_price": tl,
            "spread": spread,
            "zscore": z,
            "tc": tc,
        })

    # ------------------------------------------------------------------
    def _close_position(self, etf: float, tl: float, spread: float):
        """平仓并计算本次交易 P&L。"""
        if self.futures_pos == 0:
            return

        entry_etf = self._entry_etf_price or etf
        entry_tl = self._entry_tl_price or tl
        n_fut = self.futures_pos
        n_etf = self.etf_pos_shares

        # 期货盈亏 = 手数 × (当前价 - 开仓价) × 合约乘数
        pnl_futures = n_fut * (tl - entry_tl) * FUTURES_MULTIPLIER
        # ETF 盈亏 = 份额 × (当前价 - 开仓价)
        pnl_etf = n_etf * (etf - entry_etf)

        # 交易成本（双边）
        tc = (abs(n_etf) * etf * self.p.etf_fee_rate
              + abs(n_fut) * self.p.futures_fee_per_lot) * 2

        total = pnl_futures + pnl_etf - tc
        self.total_pnl += total

        self.trades_log.append({
            "datetime": self.data.datetime.datetime(0),
            "action": "close",
            "direction": "",
            "futures_lots": -self.futures_pos,
            "etf_shares": -int(self.etf_pos_shares),
            "etf_price": etf,
            "tl_price": tl,
            "spread": spread,
            "zscore": np.nan,
            "pnl_futures": pnl_futures,
            "pnl_etf": pnl_etf,
            "tc": tc,
            "total_pnl": total,
            "cum_pnl": self.total_pnl,
        })

        self.futures_pos = 0
        self.etf_pos_shares = 0
        self._entry_etf_price = None
        self._entry_tl_price = None

    # ------------------------------------------------------------------
    def _log_equity(self):
        self.equity_curve.append({
            "datetime": self.data.datetime.datetime(0),
            "etf": self.data.close[0],
            "tl": self.data.openinterest[0],
            "spread": self.data.high[0],
            "cum_pnl": self.total_pnl,
        })

    def stop(self):
        # 强制平仓（回测结束时仍有仓位）
        etf = self.data.close[0]
        tl = self.data.openinterest[0]
        spread = self.data.high[0]
        if self.futures_pos != 0:
            self._close_position(etf, tl, spread)
        print(f"  交易次数（开/平合计）：{len(self.trades_log)}")
        print(f"  累计 PnL：{self.total_pnl:,.2f} 元")


# ── 绩效指标 ──────────────────────────────────────────────────────────────────

def calc_performance(equity_df: pd.DataFrame,
                     initial_capital: float = 10_000_000) -> dict:
    eq = equity_df.set_index("datetime")["cum_pnl"]

    total_pnl = eq.iloc[-1]
    days = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days
    annual_return = (total_pnl / initial_capital) / max(days, 1) * 252

    daily_pnl = eq.resample("D").last().diff().dropna()
    daily_ret = daily_pnl / initial_capital
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    cum_value = initial_capital + eq
    roll_max = cum_value.expanding().max()
    max_dd = ((cum_value - roll_max) / roll_max).min()

    return {
        "绝对收益(元)": round(total_pnl, 2),
        "年化收益率(%)": round(annual_return * 100, 4),
        "夏普比率": round(sharpe, 4),
        "最大回撤(%)": round(max_dd * 100, 4),
        "交易次数": len(equity_df),
    }


# ── 主回测入口 ────────────────────────────────────────────────────────────────

def run_pairs_trading_backtest(
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    max_futures: int = 8,
    max_capital: float = 10_000_000,
    update_freq: int = 180,
) -> dict:
    """
    运行 511090 vs TL 配对交易回测（2025-07-01 ~ 2025-12-31）。
    """
    print("\n" + "=" * 60)
    print("子任务 2b：ETF-期货 OU 配对交易回测（511090 vs TL）")
    print("=" * 60)

    # 1. 构建配对数据集
    _pairs_result = build_pairs_dataset()
    pairs_df: pd.DataFrame = _pairs_result[0]
    beta: float = float(_pairs_result[1])
    n_train: int = int(_pairs_result[2])

    # 2. 初始 OU 参数（训练集估计）
    train_spread = pairs_df["spread"].values[:n_train]
    init_ou = estimate_ou_params(train_spread, dt=1.0)
    print(f"\n  初始 OU 参数（训练集）：")
    print(f"    θ (回归速度) = {init_ou['theta']:.6f} /分钟")
    print(f"    μ (长期均值) = {init_ou['mu']:.6f}")
    print(f"    σ_eq (均衡波动率) = {init_ou['sigma_eq']:.6f}")
    print(f"    半衰期 = {init_ou['half_life']:.1f} 分钟")

    # 保存初始 OU 参数
    pd.DataFrame([init_ou]).to_csv(
        OUTPUT_DIR / "2b_ou_params_initial.csv", index=False
    )

    # 3. 构造 backtrader 数据馈送
    feed_df = pairs_df[["etf_close", "tl_close", "spread"]].copy()
    feed_df["open"] = feed_df["etf_close"]
    feed_df["low"] = feed_df["etf_close"]
    feed_df.index.name = "datetime"

    data_feed = PairsData(dataname=feed_df)

    # 4. 配置 Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(
        OUPairsStrategy,
        beta=beta,
        entry_z=entry_z,
        exit_z=exit_z,
        max_futures=max_futures,
        max_capital=max_capital,
        update_freq=update_freq,
        train_bars=n_train,
        ou_window=min(1200, int(n_train)),
        init_theta=init_ou.get("theta", 0.01),
        init_mu=init_ou.get("mu", 0.0),
        init_sigma_eq=init_ou.get("sigma_eq", 1.0),
    )
    cerebro.broker.setcash(max_capital)

    # 5. 运行
    results = cerebro.run()
    strat = results[0]

    # 6. 整理结果
    equity_df = pd.DataFrame(strat.equity_curve)
    trades_df = pd.DataFrame(strat.trades_log)
    zscore_df = pd.DataFrame(strat.zscore_log)
    ou_log_df = pd.DataFrame(strat.ou_params_log)

    perf = calc_performance(equity_df, initial_capital=max_capital)

    print("\n── 绩效指标 ──")
    for k, v in perf.items():
        print(f"  {k}: {v}")

    # 7. 保存结果
    trades_df.to_csv(OUTPUT_DIR / "2b_trades.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(OUTPUT_DIR / "2b_equity.csv", index=False, encoding="utf-8-sig")
    zscore_df.to_csv(OUTPUT_DIR / "2b_zscore.csv", index=False, encoding="utf-8-sig")
    ou_log_df.to_csv(OUTPUT_DIR / "2b_ou_params_dynamic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([perf]).to_csv(OUTPUT_DIR / "2b_performance.csv", index=False, encoding="utf-8-sig")

    # 8. 画图
    _plot_pairs_results(equity_df, zscore_df, trades_df, pairs_df, n_train, perf,
                        entry_z, exit_z)

    return {"perf": perf, "equity": equity_df, "trades": trades_df,
            "zscore": zscore_df, "ou_params": init_ou}


def _plot_pairs_results(equity_df, zscore_df, trades_df, pairs_df, n_train,
                        perf, entry_z, exit_z):
    """生成配对交易结果图表（4图）。"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle("511090 ETF vs TL 期货 OU 配对交易策略", fontsize=14, fontweight="bold")

    # --- 图1：ETF 和期货价格走势 ---
    ax1 = axes[0]
    pidx = pairs_df.index.to_numpy()
    ax1.plot(pidx, pairs_df["etf_close"].to_numpy(), label="511090 ETF", lw=0.7, color="steelblue")
    ax1_r = ax1.twinx()
    ax1_r.plot(pidx, pairs_df["tl_close"].to_numpy(), label="TL期货(后复权)", lw=0.7, color="orange")
    ax1.set_title("ETF 与期货价格")
    ax1.set_ylabel("511090 价格", color="steelblue")
    ax1_r.set_ylabel("TL期货价格", color="orange")
    ax1.grid(alpha=0.3)
    # 训练/测试分割线
    if n_train < len(pairs_df):
        split_time = pairs_df.index[n_train]
        ax1.axvline(split_time, color="red", ls="--", lw=0.8, label="训练/测试分割")
    ax1.legend(loc="upper left", fontsize=7)

    # --- 图2：价差序列 ---
    ax2 = axes[1]
    ax2.plot(pidx, pairs_df["spread"].to_numpy(), color="purple", lw=0.6, label="价差")
    ax2.set_title("价差序列 (ETF - beta x Futures)")
    ax2.set_ylabel("价差")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    # --- 图3：z-score 与交易信号 ---
    ax3 = axes[2]
    if not zscore_df.empty:
        zs = zscore_df.set_index("datetime")
        zidx = zs.index.to_numpy()
        ax3.plot(zidx, zs["zscore"].to_numpy(), color="teal", lw=0.6, label="z-score")
        ax3.axhline(entry_z, color="red", ls="--", lw=0.8, label=f"入场 +{entry_z}")
        ax3.axhline(-entry_z, color="red", ls="--", lw=0.8, label=f"入场 -{entry_z}")
        ax3.axhline(exit_z, color="green", ls=":", lw=0.8, label=f"出场 +{exit_z}")
        ax3.axhline(-exit_z, color="green", ls=":", lw=0.8, label=f"出场 -{exit_z}")
        ax3.axhline(0, color="black", lw=0.4)

        # 仓位标注
        if not trades_df.empty and "action" in trades_df.columns:
            open_trades = trades_df[trades_df["action"] == "open"]
            if not open_trades.empty and "datetime" in open_trades.columns:
                for _, row in open_trades.iterrows():
                    color = "salmon" if "short" in str(row.get("direction", "")) else "lightgreen"
                    ax3.axvline(pd.to_datetime(row["datetime"]), color=color, alpha=0.3, lw=0.5)

    ax3.set_title("z-score 与入/出场信号")
    ax3.set_ylabel("z-score")
    ax3.legend(fontsize=7, ncol=3)
    ax3.grid(alpha=0.3)

    # --- 图4：累计收益曲线 ---
    ax4 = axes[3]
    if not equity_df.empty:
        eq = equity_df.set_index("datetime")
        eidx = eq.index.to_numpy()
        epnl = eq["cum_pnl"].to_numpy()
        ax4.plot(eidx, epnl, color="green", lw=1.2, label="累计收益")
        ax4.fill_between(eidx, epnl, 0,
                         where=epnl >= 0, alpha=0.2, color="green")
        ax4.fill_between(eidx, epnl, 0,
                         where=eq["cum_pnl"] < 0, alpha=0.2, color="red")
    sharpe_str = f"Sharpe={perf['夏普比率']:.2f}"
    dd_str = f"MaxDD={perf['最大回撤(%)']:.2f}%"
    ax4.set_title(f"累计收益曲线  |  {sharpe_str}  |  {dd_str}")
    ax4.set_ylabel("收益 (元)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "2b_pairs_trading_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存：{out.name}")


if __name__ == "__main__":
    run_pairs_trading_backtest()
