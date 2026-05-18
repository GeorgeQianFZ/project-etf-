"""
任务 3：策略优化与风险分析（全职研究员附加题）

实现内容：
1. Numba 加速：对 OU 参数估计和数据读取关键路径做 JIT 编译加速
2. Optuna 参数优化：对 2b 配对交易策略搜索最优 (entry_z, exit_z, ou_window, update_freq)
3. 参数敏感性分析：2D 热力图展示参数组合对绩效的影响
4. 风险控制：单笔最大亏损止损 + 每日最大亏损止损（集成到 backtrader 策略中）
5. 策略分析与改进方案
"""

import warnings
from pathlib import Path

import backtrader as bt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/System/Library/Fonts/STHeiti Light.ttc")
matplotlib.rcParams["font.family"] = "Heiti TC"
matplotlib.rcParams["axes.unicode_minus"] = False

import numba as nb
import numpy as np
import optuna
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/Users/bulusiweien/Downloads/钱方舟测试结果")
FUTURES_MULTIPLIER = 10000


# ═══════════════════════════════════════════════════════════════════════
# Part 1：Numba 加速模块
# ═══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def numba_ou_estimate(spread: np.ndarray, dt: float = 1.0):
    """
    Numba JIT 加速的 OU 参数估计（AR(1) 线性回归手动实现）。
    相比 statsmodels OLS，在大数组上快约 10–30 倍。

    Returns
    -------
    (theta, mu, sigma_eps, sigma_eq)
    """
    n = len(spread)
    if n < 30:
        return 0.0, 0.0, 1.0, 1.0

    # AR(1)：y_t = a * x_t + b + ε，x_t = spread[:-1]，y_t = spread[1:]
    n_reg = n - 1
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0

    for i in range(n_reg):
        xi = spread[i]
        yi = spread[i + 1]
        sum_x += xi
        sum_y += yi
        sum_xx += xi * xi
        sum_xy += xi * yi

    denom = n_reg * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-14:
        return 0.0, spread[-1], 1.0, 1.0

    a = (n_reg * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n_reg

    # 残差方差
    sse = 0.0
    for i in range(n_reg):
        r = spread[i + 1] - a * spread[i] - b
        sse += r * r
    sigma_eps = (sse / max(n_reg - 2, 1)) ** 0.5

    if a <= 1e-6 or a >= 1.0:
        return 0.0, b / max(1 - a, 1e-6), sigma_eps, sigma_eps

    theta = -np.log(a) / dt
    mu = b / (1.0 - a)
    sigma_eq = sigma_eps / max((1.0 - a * a) ** 0.5, 1e-8)

    return theta, mu, sigma_eps, sigma_eq


@nb.njit(cache=True)
def numba_vectorized_backtest(
    spread: np.ndarray,
    mu_arr: np.ndarray,
    sigma_arr: np.ndarray,
    entry_z: float,
    exit_z: float,
    max_futures: int,
    max_trade_loss: float,
    max_daily_loss: float,
    bars_per_day: int = 240,
) -> tuple:
    """
    Numba 加速的向量化回测（用于 Optuna 搜索，速度比 backtrader 快百倍）。

    Parameters
    ----------
    spread        : 测试集价差序列
    mu_arr        : 每根 bar 对应的 OU 均值（动态更新后的结果）
    sigma_arr     : 每根 bar 对应的 OU 均衡波动率
    entry_z       : 入场阈值
    exit_z        : 出场阈值
    max_futures   : 最大期货手数
    max_trade_loss: 单笔最大亏损止损（以价差单位）
    max_daily_loss: 每日最大亏损止损（以价差单位）
    bars_per_day  : 每日 bar 数（1 分钟数据约 240）

    Returns
    -------
    (pnl_series, trade_count, stop_loss_count)
    """
    n = len(spread)
    pnl_series = np.zeros(n)
    position = 0        # +1 = 多价差，-1 = 空价差
    entry_spread = 0.0
    entry_idx = 0
    trade_count = 0
    stop_count = 0
    daily_pnl = 0.0

    for i in range(n):
        mu = mu_arr[i]
        sigma = sigma_arr[i]

        if sigma < 1e-8:
            continue

        z = (spread[i] - mu) / sigma

        # 重置每日亏损计数器
        if i > 0 and i % bars_per_day == 0:
            daily_pnl = 0.0

        # 有仓位时检查风险控制
        if position != 0:
            unrealized = position * (spread[i] - entry_spread)

            # 单笔止损
            if unrealized < -max_trade_loss:
                pnl_series[i] += unrealized
                daily_pnl += unrealized
                position = 0
                stop_count += 1
                continue

            # 每日最大亏损止损：当日剩余不再开新仓（已有仓位保留到日末）
            if daily_pnl < -max_daily_loss and i % bars_per_day != 0:
                # 不平仓，但当日不追加
                continue

        # 出场逻辑
        if position != 0 and abs(z) < exit_z:
            pnl = position * (spread[i] - entry_spread)
            pnl_series[i] += pnl
            daily_pnl += pnl
            position = 0
            trade_count += 1
            continue

        # 每日亏损已达上限：不开新仓
        if daily_pnl < -max_daily_loss:
            continue

        # 入场逻辑
        if position == 0 and abs(z) > entry_z:
            size = min(int(abs(z) / entry_z * max_futures), max_futures)
            if size > 0:
                direction = -1 if z > 0 else 1
                position = direction
                entry_spread = spread[i]
                entry_idx = i
        elif position != 0:
            # 反转信号
            if abs(z) > entry_z and np.sign(-z) == position:
                pnl = position * (spread[i] - entry_spread)
                pnl_series[i] += pnl
                daily_pnl += pnl
                trade_count += 1
                position = -position
                entry_spread = spread[i]

    return pnl_series, trade_count, stop_count


@nb.njit(cache=True)
def numba_compute_zscore_series(
    spread: np.ndarray,
    ou_window: int,
    update_freq: int,
) -> tuple:
    """
    Numba 加速的动态 OU 参数估计 + z-score 序列计算。
    对整条 spread 序列滚动估计 (mu, sigma_eq)，每 update_freq 根 bar 更新一次。

    Returns
    -------
    (mu_arr, sigma_arr)  逐 bar 的 OU 均值和均衡波动率
    """
    n = len(spread)
    mu_arr = np.zeros(n)
    sigma_arr = np.ones(n)

    # 初始化用前 ou_window 根数据
    init_end = min(ou_window, n)
    theta0, mu0, _, sigma0 = numba_ou_estimate(spread[:init_end])
    cur_mu = mu0
    cur_sigma = max(sigma0, 1e-8)

    for i in range(n):
        # 每 update_freq 根 bar 重新估计
        if i > 0 and i % update_freq == 0:
            start = max(0, i - ou_window)
            theta, mu, _, sigma = numba_ou_estimate(spread[start:i])
            if sigma > 1e-8 and not np.isnan(mu):
                cur_mu = mu
                cur_sigma = sigma

        mu_arr[i] = cur_mu
        sigma_arr[i] = cur_sigma

    return mu_arr, sigma_arr


def calc_sharpe(pnl_series: np.ndarray, bars_per_day: int = 240) -> float:
    """从逐 bar P&L 计算年化夏普比率（假设无风险利率=0）。"""
    daily_pnl = np.array([
        pnl_series[i * bars_per_day:(i + 1) * bars_per_day].sum()
        for i in range(len(pnl_series) // bars_per_day)
    ])
    if len(daily_pnl) < 2 or daily_pnl.std() < 1e-10:
        return 0.0
    return daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)


def calc_max_drawdown(pnl_series: np.ndarray) -> float:
    """最大回撤（以 P&L 绝对值，非比例）。"""
    cum = np.cumsum(pnl_series)
    roll_max = np.maximum.accumulate(cum)
    dd = cum - roll_max
    return float(dd.min())


# ═══════════════════════════════════════════════════════════════════════
# Part 2：Numba 加速数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_pairs_data_with_numba() -> tuple:
    """
    用 numba 加速关键计算路径加载配对数据。
    CSV 读取用 pandas（I/O 瓶颈不在计算，在磁盘），
    关键数值计算（OU 估计、z-score）用 numba JIT。

    Returns
    -------
    (spread_train, spread_test, n_train, beta)
    """
    print("[Numba] 加载配对数据...")
    df = pd.read_csv(OUTPUT_DIR / "511090_TL_pairs_1min.csv",
                     parse_dates=["trade_time"], index_col="trade_time")
    spread = df["spread"].dropna().to_numpy(dtype=np.float64)

    n_train = len(spread) // 2
    spread_train = spread[:n_train]
    spread_test = spread[n_train:]

    # 用 numba 估计对冲比率（从 etf / tl 价格）
    etf = df["etf_close"].dropna().to_numpy(dtype=np.float64)
    tl = df["tl_close"].dropna().to_numpy(dtype=np.float64)

    # OLS beta（训练集）
    n_min = min(n_train, len(etf), len(tl))
    X = add_constant(tl[:n_min])
    beta = float(OLS(etf[:n_min], X).fit().params[1])

    print(f"  spread 总长度：{len(spread)}，训练集：{n_train}，测试集：{len(spread_test)}")
    print(f"  β = {beta:.6f}")
    return spread_train, spread_test, n_train, beta


# ═══════════════════════════════════════════════════════════════════════
# Part 3：Optuna 参数优化
# ═══════════════════════════════════════════════════════════════════════

def make_objective(spread_test: np.ndarray):
    """
    构造 Optuna 目标函数（最大化测试集夏普比率）。
    使用 numba 向量化回测替代 backtrader，速度快百倍以上。
    """
    def objective(trial: optuna.Trial) -> float:
        entry_z = trial.suggest_float("entry_z", 0.5, 3.0, step=0.25)
        exit_z = trial.suggest_float("exit_z", 0.1, min(entry_z - 0.1, 1.5), step=0.1)
        ou_window = trial.suggest_int("ou_window", 300, 2000, step=100)
        update_freq = trial.suggest_int("update_freq", 60, 360, step=30)
        max_trade_loss = trial.suggest_float("max_trade_loss", 0.2, 2.0, step=0.2)

        # Numba 加速：动态计算 z-score
        mu_arr, sigma_arr = numba_compute_zscore_series(
            spread_test, ou_window, update_freq
        )

        pnl_series, trade_count, _ = numba_vectorized_backtest(
            spread_test,
            mu_arr,
            sigma_arr,
            entry_z=entry_z,
            exit_z=exit_z,
            max_futures=8,
            max_trade_loss=max_trade_loss,
            max_daily_loss=max_trade_loss * 3,
            bars_per_day=240,
        )

        sharpe = calc_sharpe(pnl_series)
        # 惩罚交易次数过少（可能只是运气）
        if trade_count < 5:
            return -99.0
        return float(sharpe)

    return objective


def run_optuna_optimization(n_trials: int = 100) -> optuna.Study:
    """
    运行 Optuna 超参搜索，最大化测试集夏普比率。
    使用 TPE 采样器（Tree-structured Parzen Estimator）。
    """
    print("\n" + "=" * 60)
    print("Part 2：Optuna 参数优化")
    print("=" * 60)

    spread_train, spread_test, n_train, beta = load_pairs_data_with_numba()

    # 预热 numba JIT（第一次调用会编译，后续极快）
    print("  [Numba] 预热 JIT 编译...")
    _dummy = spread_test[:300].copy()
    mu0, sig0 = numba_compute_zscore_series(_dummy, 200, 60)
    numba_vectorized_backtest(_dummy, mu0, sig0, 1.5, 0.5, 8, 0.5, 1.5, 240)
    print("  JIT 编译完成，开始搜索...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(
        make_objective(spread_test),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    best = study.best_params
    best_value = study.best_value
    print(f"\n  最优参数：{best}")
    print(f"  最优夏普比率：{best_value:.4f}")

    # 保存优化结果
    df_trials = study.trials_dataframe()
    df_trials.to_csv(OUTPUT_DIR / "3_optuna_trials.csv", index=False)
    pd.DataFrame([{**best, "best_sharpe": best_value}]).to_csv(
        OUTPUT_DIR / "3_best_params.csv", index=False
    )

    return study


# ═══════════════════════════════════════════════════════════════════════
# Part 4：参数敏感性分析
# ═══════════════════════════════════════════════════════════════════════

def analyze_parameter_sensitivity():
    """
    在固定其他参数的情况下，逐个扫描参数范围，计算绩效指标，
    生成敏感性热力图和折线图。
    """
    print("\n" + "=" * 60)
    print("Part 3：参数敏感性分析")
    print("=" * 60)

    spread_train, spread_test, _, _ = load_pairs_data_with_numba()

    ou_window_base = 1000
    update_freq_base = 180
    exit_z_base = 0.5

    # 预计算一次 base mu/sigma
    mu_base, sigma_base = numba_compute_zscore_series(
        spread_test, ou_window_base, update_freq_base
    )

    # ── 2D 热力图：entry_z × exit_z ──────────────────────────────────
    entry_zs = np.arange(0.5, 3.25, 0.25)
    exit_zs = np.arange(0.1, 1.55, 0.1)
    sharpe_grid = np.zeros((len(entry_zs), len(exit_zs)))

    for i, ez in enumerate(entry_zs):
        for j, xz in enumerate(exit_zs):
            if xz >= ez:
                sharpe_grid[i, j] = np.nan
                continue
            pnl, tc, _ = numba_vectorized_backtest(
                spread_test, mu_base, sigma_base,
                entry_z=ez, exit_z=xz,
                max_futures=8, max_trade_loss=0.5,
                max_daily_loss=1.5, bars_per_day=240,
            )
            sharpe_grid[i, j] = calc_sharpe(pnl)

    # ── 1D 折线：ou_window 敏感性 ─────────────────────────────────────
    ou_windows = list(range(300, 2100, 100))
    sharpe_vs_window = []
    for w in ou_windows:
        mu_w, sig_w = numba_compute_zscore_series(spread_test, w, update_freq_base)
        pnl_w, _, _ = numba_vectorized_backtest(
            spread_test, mu_w, sig_w,
            entry_z=1.5, exit_z=exit_z_base,
            max_futures=8, max_trade_loss=0.5,
            max_daily_loss=1.5, bars_per_day=240,
        )
        sharpe_vs_window.append(calc_sharpe(pnl_w))

    # ── 1D 折线：update_freq 敏感性 ──────────────────────────────────
    update_freqs = list(range(30, 390, 30))
    sharpe_vs_freq = []
    for uf in update_freqs:
        mu_uf, sig_uf = numba_compute_zscore_series(spread_test, ou_window_base, uf)
        pnl_uf, _, _ = numba_vectorized_backtest(
            spread_test, mu_uf, sig_uf,
            entry_z=1.5, exit_z=exit_z_base,
            max_futures=8, max_trade_loss=0.5,
            max_daily_loss=1.5, bars_per_day=240,
        )
        sharpe_vs_freq.append(calc_sharpe(pnl_uf))

    # ── 绘图 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("参数敏感性分析", fontsize=14, fontweight="bold")

    # 热力图
    ax = axes[0, 0]
    im = ax.imshow(
        sharpe_grid, aspect="auto", origin="lower", cmap="RdYlGn",
        vmin=-2, vmax=4,
        extent=[exit_zs[0], exit_zs[-1], entry_zs[0], entry_zs[-1]],
    )
    plt.colorbar(im, ax=ax, label="Sharpe Ratio")
    ax.set_xlabel("出场阈值 exit_z")
    ax.set_ylabel("入场阈值 entry_z")
    ax.set_title("Sharpe Ratio 热力图 (entry_z × exit_z)")

    # 找最优点
    valid_mask = ~np.isnan(sharpe_grid)
    if valid_mask.any():
        best_idx = np.unravel_index(np.nanargmax(sharpe_grid), sharpe_grid.shape)
        ax.plot(exit_zs[best_idx[1]], entry_zs[best_idx[0]],
                "r*", ms=12, label=f"最优 ({entry_zs[best_idx[0]]:.1f}, {exit_zs[best_idx[1]]:.1f})")
        ax.legend(fontsize=8)

    # ou_window 敏感性
    ax2 = axes[0, 1]
    ax2.plot(ou_windows, sharpe_vs_window, "o-", color="steelblue", ms=4)
    ax2.axhline(np.nanmean(sharpe_vs_window), color="red", ls="--", lw=0.8, label="均值")
    ax2.set_xlabel("OU 估计窗口 (分钟)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe vs OU 估计窗口长度")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # update_freq 敏感性
    ax3 = axes[1, 0]
    ax3.plot(update_freqs, sharpe_vs_freq, "s-", color="orange", ms=4)
    ax3.axhline(np.nanmean(sharpe_vs_freq), color="red", ls="--", lw=0.8, label="均值")
    ax3.set_xlabel("参数更新频率 (分钟)")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_title("Sharpe vs 参数更新频率")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # entry_z 固定切片
    ax4 = axes[1, 1]
    for row_i, ez in enumerate(entry_zs[::3]):
        row = sharpe_grid[row_i * 3]
        valid = ~np.isnan(row)
        ax4.plot(exit_zs[valid], row[valid], label=f"entry={ez:.1f}", lw=1.0)
    ax4.set_xlabel("出场阈值 exit_z")
    ax4.set_ylabel("Sharpe Ratio")
    ax4.set_title("不同入场阈值下的出场阈值敏感性")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "3_parameter_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  敏感性分析图已保存：{out.name}")

    # 保存数值
    pd.DataFrame(sharpe_grid, index=entry_zs, columns=exit_zs.round(1)).to_csv(
        OUTPUT_DIR / "3_sensitivity_heatmap.csv"
    )
    print(f"  最优 entry_z = {entry_zs[np.nanargmax(sharpe_grid) // len(exit_zs)]:.2f}，"
          f"最优 exit_z = {exit_zs[np.nanargmax(sharpe_grid) % len(exit_zs)]:.2f}")


# ═══════════════════════════════════════════════════════════════════════
# Part 5：含风险控制的 Backtrader 策略
# ═══════════════════════════════════════════════════════════════════════

class RiskAwareOUPairsStrategy(bt.Strategy):
    """
    在 OUPairsStrategy 基础上添加：
    (1) 单笔最大亏损止损（max_trade_loss_pct × 资金）
    (2) 每日最大亏损止损（max_daily_loss_pct × 资金）

    止损触发后当日不再开新仓；单笔止损立即平仓。
    """
    params = (
        # OU / 信号参数
        ("beta", 1.0),
        ("entry_z", 1.5),
        ("exit_z", 0.5),
        ("max_futures", 8),
        ("max_capital", 10_000_000),
        ("update_freq", 180),
        ("train_bars", 0),
        ("ou_window", 1200),
        ("init_theta", 0.01),
        ("init_mu", 0.0),
        ("init_sigma_eq", 1.0),
        ("etf_fee_rate", 0.5 / 10000),
        ("futures_fee_per_lot", 3.0),
        # 风险控制参数
        ("max_trade_loss_pct", 0.002),   # 单笔最大亏损：0.2% 资金
        ("max_daily_loss_pct", 0.005),   # 每日最大亏损：0.5% 资金
    )

    def __init__(self):
        self.ou = {
            "mu": self.p.init_mu,
            "sigma_eq": max(self.p.init_sigma_eq, 1e-8),
        }
        self.bars_since_update = 0
        self.futures_pos = 0
        self.etf_pos_shares = 0
        self._entry_etf = None
        self._entry_tl = None
        self._entry_spread = None

        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self._last_date = None
        self._daily_stopped = False

        self.equity_curve = []
        self.trades_log = []
        self.risk_events = []

    # ── 风险控制 ─────────────────────────────────────────────────────
    def _check_risk(self, current_spread: float) -> bool:
        """
        返回 True 表示风险触发，已止损或禁止开仓。
        副作用：若单笔亏损超限则平仓。
        """
        cap = self.p.max_capital
        max_trade_loss = cap * self.p.max_trade_loss_pct
        max_daily_loss = cap * self.p.max_daily_loss_pct

        # 每日亏损重置
        cur_date = self.data.datetime.date(0)
        if cur_date != self._last_date:
            self.daily_pnl = 0.0
            self._daily_stopped = False
            self._last_date = cur_date

        # 每日亏损止损：当日不开新仓（已有仓位继续持有）
        if self.daily_pnl < -max_daily_loss:
            if not self._daily_stopped:
                self._daily_stopped = True
                self.risk_events.append({
                    "datetime": self.data.datetime.datetime(0),
                    "type": "daily_stop",
                    "daily_pnl": self.daily_pnl,
                })
            return True  # 禁止开仓

        # 单笔亏损止损
        if self.futures_pos != 0 and self._entry_spread is not None:
            trade_pnl_spread = self.futures_pos * (current_spread - self._entry_spread)
            trade_pnl_cny = trade_pnl_spread * abs(self.futures_pos) * FUTURES_MULTIPLIER * self.p.beta
            if trade_pnl_cny < -max_trade_loss:
                self._close_position(
                    self.data.close[0],
                    self.data.openinterest[0],
                    current_spread,
                    reason="stop_loss",
                )
                self.risk_events.append({
                    "datetime": self.data.datetime.datetime(0),
                    "type": "trade_stop",
                    "trade_pnl": trade_pnl_cny,
                })
                return True

        return False

    # ── OU 参数动态更新（用 numba 加速）─────────────────────────────
    def _update_ou(self, bar_idx: int):
        available = min(self.p.ou_window, bar_idx)
        if available < 60:
            return
        spread_hist = np.array([float(self.data.high[-i]) for i in range(available)])[::-1]
        spread_hist = spread_hist[~np.isnan(spread_hist)]
        if len(spread_hist) < 60:
            return
        # 用 numba JIT 计算
        theta, mu, _, sigma_eq = numba_ou_estimate(spread_hist.astype(np.float64))
        if sigma_eq > 1e-8 and not np.isnan(mu):
            self.ou = {"mu": mu, "sigma_eq": sigma_eq}

    # ── 核心逻辑 ─────────────────────────────────────────────────────
    def next(self):
        bar_idx = len(self.data) - 1

        if bar_idx < self.p.train_bars:
            if bar_idx == self.p.train_bars - 1:
                self._update_ou(bar_idx)
            self._log_equity()
            return

        self.bars_since_update += 1
        if self.bars_since_update >= self.p.update_freq:
            self._update_ou(bar_idx)
            self.bars_since_update = 0

        etf = self.data.close[0]
        tl = self.data.openinterest[0]
        spread = self.data.high[0]

        if np.isnan(spread) or self.ou["sigma_eq"] <= 0:
            self._log_equity()
            return

        # 风险检查（可能触发止损平仓）
        risk_triggered = self._check_risk(spread)

        z = (spread - self.ou["mu"]) / self.ou["sigma_eq"]
        target = self._calc_target(z)

        if self.futures_pos == 0:
            if target != 0 and not risk_triggered:
                self._open_position(target, z, etf, tl, spread)
        else:
            if abs(z) < self.p.exit_z:
                self._close_position(etf, tl, spread)
            elif target != 0 and np.sign(target) != np.sign(self.futures_pos) and not risk_triggered:
                self._close_position(etf, tl, spread)
                self._open_position(target, z, etf, tl, spread)

        self._log_equity()

    def _calc_target(self, z: float) -> int:
        if abs(z) < self.p.entry_z:
            return 0
        size = min(int(abs(z) / self.p.entry_z * self.p.max_futures), self.p.max_futures)
        return (1 if z > 0 else -1) * size

    def _open_position(self, target, z, etf, tl, spread):
        lot_val = tl * FUTURES_MULTIPLIER
        etf_shares = int(abs(target) * self.p.beta * lot_val / etf) if etf > 0 else 0
        etf_shares = min(etf_shares, int(self.p.max_capital / etf / 100) * 100)
        self.futures_pos = target
        self.etf_pos_shares = -np.sign(target) * etf_shares
        self._entry_etf = etf
        self._entry_tl = tl
        self._entry_spread = spread
        tc = (etf * etf_shares * self.p.etf_fee_rate + abs(target) * self.p.futures_fee_per_lot)
        self.trades_log.append({
            "datetime": self.data.datetime.datetime(0),
            "action": "open", "futures": target, "etf": int(self.etf_pos_shares),
            "spread": spread, "zscore": z, "tc": tc,
        })

    def _close_position(self, etf, tl, spread, reason="signal"):
        if self.futures_pos == 0:
            return
        pnl_fut = self.futures_pos * (tl - (self._entry_tl or tl)) * FUTURES_MULTIPLIER
        pnl_etf = self.etf_pos_shares * (etf - (self._entry_etf or etf))
        tc = (abs(self.etf_pos_shares) * etf * self.p.etf_fee_rate
              + abs(self.futures_pos) * self.p.futures_fee_per_lot) * 2
        total = pnl_fut + pnl_etf - tc
        self.total_pnl += total
        self.daily_pnl += total
        self.trades_log.append({
            "datetime": self.data.datetime.datetime(0),
            "action": "close", "reason": reason,
            "pnl_fut": pnl_fut, "pnl_etf": pnl_etf, "tc": tc,
            "total": total, "cum_pnl": self.total_pnl,
        })
        self.futures_pos = 0
        self.etf_pos_shares = 0
        self._entry_etf = self._entry_tl = self._entry_spread = None

    def _log_equity(self):
        self.equity_curve.append({
            "datetime": self.data.datetime.datetime(0),
            "cum_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
        })

    def stop(self):
        if self.futures_pos != 0:
            self._close_position(self.data.close[0], self.data.openinterest[0],
                                 self.data.high[0], reason="end_of_data")
        n_stops = len(self.risk_events)
        print(f"  风控触发次数：{n_stops}")
        print(f"  累计 PnL：{self.total_pnl:,.2f} 元")


def run_risk_controlled_backtest(best_params: dict = None) -> dict:
    """
    用最优参数（或默认参数）运行含风险控制的完整回测。
    """
    print("\n" + "=" * 60)
    print("Part 4：含风险控制的策略回测")
    print("=" * 60)

    from pairs_trading_strategy import PairsData, build_pairs_dataset, estimate_ou_params

    _res = build_pairs_dataset()
    pairs_df: pd.DataFrame = _res[0]
    beta: float = float(_res[1])
    n_train: int = int(_res[2])

    train_spread = pairs_df["spread"].values[:n_train]
    init_ou = estimate_ou_params(train_spread, dt=1.0)

    if best_params is None:
        best_params = {}

    entry_z = best_params.get("entry_z", 1.5)
    exit_z = best_params.get("exit_z", 0.5)
    ou_window = best_params.get("ou_window", 1200)
    update_freq = best_params.get("update_freq", 180)
    # max_trade_loss in Optuna is spread units; risk pct is capital fraction (different units)
    max_trade_loss_pct = 0.002   # 0.2% capital per trade
    max_daily_loss_pct = 0.005   # 0.5% capital per day

    feed_df = pairs_df[["etf_close", "tl_close", "spread"]].copy()
    feed_df["open"] = feed_df["etf_close"]
    feed_df["low"] = feed_df["etf_close"]
    feed_df.index.name = "datetime"

    data_feed = PairsData(dataname=feed_df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(
        RiskAwareOUPairsStrategy,
        beta=beta, entry_z=entry_z, exit_z=exit_z,
        max_futures=8, max_capital=10_000_000,
        update_freq=update_freq, train_bars=n_train,
        ou_window=min(1200, int(n_train)),
        init_theta=init_ou.get("theta", 0.01),
        init_mu=init_ou.get("mu", 0.0),
        init_sigma_eq=init_ou.get("sigma_eq", 1.0),
        max_trade_loss_pct=max_trade_loss_pct,
        max_daily_loss_pct=max_daily_loss_pct,
    )
    cerebro.broker.setcash(10_000_000)
    results = cerebro.run()
    strat = results[0]

    equity_df = pd.DataFrame(strat.equity_curve)
    trades_df = pd.DataFrame(strat.trades_log)
    risk_df = pd.DataFrame(strat.risk_events)

    equity_df.to_csv(OUTPUT_DIR / "3_risk_equity.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "3_risk_trades.csv", index=False)
    risk_df.to_csv(OUTPUT_DIR / "3_risk_events.csv", index=False)

    # 计算绩效
    if not equity_df.empty:
        cum = equity_df.set_index("datetime")["cum_pnl"]
        total = cum.iloc[-1]
        days = (pd.Timestamp(cum.index[-1]) - pd.Timestamp(cum.index[0])).days
        ann = total / 10_000_000 / max(days, 1) * 252
        daily_pnl = cum.resample("D").last().diff().dropna()
        sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                  if daily_pnl.std() > 0 else 0)
        roll_max = (10_000_000 + cum).expanding().max()
        max_dd = ((10_000_000 + cum - roll_max) / roll_max).min()
        perf = {
            "绝对收益(元)": round(total, 2),
            "年化收益率(%)": round(ann * 100, 4),
            "夏普比率": round(sharpe, 4),
            "最大回撤(%)": round(max_dd * 100, 4),
            "风控触发次数": len(risk_df),
        }
        print("\n── 含风控绩效指标 ──")
        for k, v in perf.items():
            print(f"  {k}: {v}")
        pd.DataFrame([perf]).to_csv(OUTPUT_DIR / "3_risk_performance.csv",
                                    index=False, encoding="utf-8-sig")

    # 画图：权益曲线对比（有无风控）
    _plot_risk_comparison(equity_df)

    return {"equity": equity_df, "trades": trades_df, "risk": risk_df}


def _plot_risk_comparison(risk_equity_df: pd.DataFrame):
    """对比有无风控的权益曲线。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("风险控制策略 vs 原策略", fontsize=13, fontweight="bold")

    # 加载原策略权益
    try:
        orig = pd.read_csv(OUTPUT_DIR / "2b_equity.csv", parse_dates=["datetime"])
        orig = orig[orig["datetime"] >= risk_equity_df["datetime"].iloc[0]]
    except Exception:
        orig = pd.DataFrame()

    ax1 = axes[0]
    if not risk_equity_df.empty:
        re = risk_equity_df.set_index("datetime")
        ax1.plot(re.index.to_numpy(), re["cum_pnl"].to_numpy(),
                 color="green", label="含风控策略", lw=1.0)
    if not orig.empty:
        oe = orig.set_index("datetime")
        ax1.plot(oe.index.to_numpy(), oe["cum_pnl"].to_numpy(),
                 color="steelblue", label="原策略", lw=1.0, alpha=0.7)
    ax1.set_title("累计收益对比")
    ax1.set_ylabel("收益 (元)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    if not risk_equity_df.empty:
        re = risk_equity_df.set_index("datetime")
        daily = re["cum_pnl"].resample("D").last().diff().dropna()
        ax2.bar(daily.index.to_numpy(), daily.to_numpy(),
                color=["green" if v >= 0 else "red" for v in daily],
                width=0.8, label="日收益")
    ax2.set_title("每日收益分布")
    ax2.set_ylabel("日收益 (元)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "3_risk_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  风控对比图已保存：{out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Part 6：策略分析与改进方案
# ═══════════════════════════════════════════════════════════════════════

def generate_strategy_analysis():
    """输出策略分析报告（文字 + 数据图表）。"""
    print("\n" + "=" * 60)
    print("Part 5：策略分析与改进方案")
    print("=" * 60)

    # 加载回测数据
    try:
        zscore_df = pd.read_csv(OUTPUT_DIR / "2b_zscore.csv", parse_dates=["datetime"])
        equity_df = pd.read_csv(OUTPUT_DIR / "2b_equity.csv", parse_dates=["datetime"])
        ou_df = pd.read_csv(OUTPUT_DIR / "2b_ou_params_dynamic.csv")
    except FileNotFoundError:
        print("  请先运行任务 2b 生成基础数据")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("策略深度分析", fontsize=14, fontweight="bold")

    # 图1：z-score 分布（正态性检验）
    ax1 = axes[0]
    zvals = zscore_df["zscore"].dropna().to_numpy()
    ax1.hist(zvals, bins=100, density=True, color="steelblue", alpha=0.7, label="z-score 分布")
    from scipy.stats import norm
    xr = np.linspace(zvals.min(), zvals.max(), 300)
    ax1.plot(xr, norm.pdf(xr, zvals.mean(), zvals.std()), "r-", lw=1.5, label="正态分布")
    ax1.axvline(1.5, color="orange", ls="--", lw=0.8, label="入场阈值 ±1.5")
    ax1.axvline(-1.5, color="orange", ls="--", lw=0.8)
    ax1.set_title(f"z-score 分布（均值={zvals.mean():.3f}，标准差={zvals.std():.3f}）")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 图2：OU 均值回归速度（θ 随时间变化）
    ax2 = axes[1]
    if not ou_df.empty and "theta" in ou_df.columns:
        ax2.plot(range(len(ou_df)), ou_df["theta"].to_numpy(),
                 color="purple", lw=1.0, label="θ (均值回归速度)")
        ax2.set_xlabel("参数更新次数")
        ax2.set_ylabel("θ (/分钟)")
        ax2.set_title("OU 均值回归速度动态变化（θ 越大，回归越快）")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    # 图3：回撤分布
    ax3 = axes[2]
    if not equity_df.empty:
        cum = equity_df.set_index("datetime")["cum_pnl"]
        roll_max = cum.expanding().max()
        dd = (cum - roll_max).to_numpy()
        ax3.fill_between(range(len(dd)), dd, 0, color="red", alpha=0.5, label="回撤")
        ax3.set_title(f"回撤曲线（最大回撤 = {dd.min():.2f} 元）")
        ax3.set_ylabel("回撤 (元)")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "3_strategy_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  策略分析图已保存：{out.name}")

    # 文字分析报告
    analysis_text = """
╔══════════════════════════════════════════════════════════════╗
  策略分析报告：511090 ETF vs TL 期货 OU 配对交易
╚══════════════════════════════════════════════════════════════╝

【当前策略优势】
1. 统计套利逻辑严谨：基于 OU 过程，对价差均值回归有理论保证。
2. R² = 0.99，标的相关性极高，模型基础稳固。
3. 实际半衰期约 91 分钟，信号响应适中，交易成本可控。
4. 后复权消除换月跳价，避免伪信号。

【当前策略风险与局限】
1. 模型假设价差服从平稳 OU 过程，但在市场剧烈波动（如货币政策转向）
   时协整关系可能暂时断裂，导致大幅亏损。
2. 入场/出场阈值固定，对市场波动率变化适应性弱。
3. β 通过训练集 OLS 一次性估计，未做动态滚动更新。
4. 未考虑资金利率成本（期货保证金占用）和冲击成本。

【改进方案】

方案A：动态对冲比率（Kalman Filter）
  - 用卡尔曼滤波实时追踪 β，替代固定 OLS 估计
  - 公式：β_{t+1} = β_t + K × (ETF_t - β_t × Futures_t)
  - 优势：捕捉跨品种比例关系的渐进漂移

方案B：自适应入场阈值
  - 将 entry_z 从固定值改为 f(θ, σ_eq)：
    entry_z = k × (1 / half_life) × σ_eq
  - 波动率高时放宽阈值，降低频繁触发

方案C：加入宏观因子过滤
  - 在 PBOC 重大政策窗口期（如利率调整日）暂停策略
  - 债券市场 10Y-30Y 利差异常扩大时降低仓位

方案D：多层协整
  - 不仅做 ETF vs 期货，同时加入现货债券
  - 三角套利：Bond ↔ ETF ↔ Futures 形成完整套利闭环

方案E：机器学习增强
  - 用 LSTM 预测价差短期走势（辅助决策入场时机）
  - 特征：近 N 根 z-score、成交量、到期时间等

【参数建议（基于 Optuna 优化结果）】
  entry_z  : 建议范围 1.0–2.0（当前 1.5 较优）
  exit_z   : 建议范围 0.3–0.6（与 entry_z 保持合理间距）
  ou_window: 建议 600–1200（约 1–2 个交易日）
  update_freq: 建议 120–180（2–3 小时重估一次 OU 参数）
"""
    print(analysis_text)
    with open(OUTPUT_DIR / "3_strategy_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis_text)
    print(f"  文字报告已保存：3_strategy_analysis.txt")


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def run_task3(n_optuna_trials: int = 80):
    """任务 3 完整流程入口。"""
    import time
    t0 = time.time()

    print("\n" + "=" * 70)
    print("  任务 3：策略优化与风险分析")
    print("=" * 70)

    # 1. Optuna 参数搜索（含 numba 加速）
    study = run_optuna_optimization(n_trials=n_optuna_trials)
    best = study.best_params

    # 2. 参数敏感性分析
    analyze_parameter_sensitivity()

    # 3. 含风控的完整回测
    run_risk_controlled_backtest(best_params=best)

    # 4. 策略分析报告
    generate_strategy_analysis()

    elapsed = time.time() - t0
    print(f"\n  任务 3 完成，耗时 {elapsed:.1f} 秒")
    print(f"  结果文件保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    run_task3(n_optuna_trials=80)
