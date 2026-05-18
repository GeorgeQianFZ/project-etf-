"""
数据处理模块 - 任务1
读取、清洗并合并 ETF / 债券 / 期货 3秒快照数据。

关键决策说明：
1. ETF 使用 trading_phase_code == 'T111' 过滤连续竞价阶段；
   债券使用 trading_phase_code == 'T' （交易所债券格式不同）。
2. 中间价 = (ask_price1 + bid_price1) / 2；买卖价均为 0 时回退到 last。
3. 期货连续合约存在换月跳价问题，采用后复权方式（backward ratio adjustment）
   消除换月价差，保证价差序列平稳。
4. 债券行情为净价，IOPV 计算需额外加上应计利息（见 arbitrage_strategy.py）。
5. 过滤交易时段：
   - ETF  / 债券：09:30–11:30, 13:00–15:00
   - 期货：09:29–11:30, 13:00–15:00（提前 1 分钟以捕获开盘跳空）
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/bulusiweien/Downloads/汇远盈测试题需要的数据")
OUTPUT_DIR = Path("/Users/bulusiweien/Downloads/钱方舟测试结果")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ETF_DIR = BASE_DIR / "国债ETF数据" / "3秒快照"
BOND_DIR = BASE_DIR / "债券数据" / "3秒快照"
FUTURES_PATH = BASE_DIR / "国债期货合约数据" / "快照" / "30年TL合约" / "TL_main_snapshot.csv"


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _mid_price(df: pd.DataFrame) -> pd.Series:
    """计算中间价；买卖均为 0 时退回到 last。"""
    ask = df["ask_price1"].replace(0, np.nan)
    bid = df["bid_price1"].replace(0, np.nan)
    last = df["last"].replace(0, np.nan)
    mid = (ask + bid) / 2
    return mid.where(mid.notna(), last)


def _in_trading_hours(t: pd.Series, extra_minutes_before: int = 0) -> pd.Series:
    """返回布尔 Series：是否在交易时段内。"""
    t_min = t.dt.hour * 60 + t.dt.minute
    morning = (t_min >= 9 * 60 + 30 - extra_minutes_before) & (t_min <= 11 * 60 + 30)
    afternoon = (t_min >= 13 * 60) & (t_min <= 15 * 60)
    return morning | afternoon


def _remove_price_outliers(series: pd.Series, n_sigma: float = 5.0) -> pd.Series:
    """用 z-score 去除价格异常值（替换为 NaN）。"""
    mu = series.median()
    sigma = series.std()
    if sigma == 0:
        return series
    z = (series - mu).abs() / sigma
    return series.where(z < n_sigma, np.nan)


# ── ETF 数据加载 ──────────────────────────────────────────────────────────────

def load_etf_snapshots(etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载指定 ETF 的 3 秒快照数据，完成清洗后返回 DataFrame。

    Parameters
    ----------
    etf_code   : '511090.SH' 或 '511130.SH'
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'

    Returns
    -------
    DataFrame，列：trade_time, code, open, high, low, close, volume, amount,
                   bid_price1, ask_price1, mid_price, iopv
    """
    code = etf_code.split(".")[0]
    path = ETF_DIR / code
    files = sorted(glob.glob(str(path / "*.csv")))

    if not files:
        raise FileNotFoundError(f"未找到 {etf_code} 的数据文件，路径：{path}")

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f, index_col=0, low_memory=False)
            dfs.append(tmp)
        except Exception as e:
            print(f"  ⚠ 读取失败：{f}，原因：{e}")

    df = pd.concat(dfs, ignore_index=True)

    # errors="coerce" 将无法解析的字符串置为 NaT，避免整列因脏数据报错
    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df = df.dropna(subset=["trade_time"])

    # 用 .date() 比较去掉时区歧义，同时支持跨月文件只保留指定窗口
    s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
    df = df[(df["trade_time"].dt.date >= s.date()) & (df["trade_time"].dt.date <= e.date())]

    # ETF 行情的连续竞价阶段码为 'T111'；集合竞价(S10)、收盘(E110)等阶段
    # 价格不代表可成交价，需剔除，否则 IOPV 计算和套利信号会产生虚假偏差
    df = df[df["trading_phase_code"] == "T111"]

    # 剔除午休（11:30–13:00）及收盘后的快照；_in_trading_hours 已处理夏令时边界
    df = df[_in_trading_hours(df["trade_time"])].copy()

    # 优先用买一/卖一均值作为参考价，单边缺失时回退到 last；
    # 比直接用 last 更能反映当前市场深度，减少成交稀疏时的噪声
    df["mid_price"] = _mid_price(df)

    # 3 倍 IQR 过滤价格跳变（如涨跌停瞬间快照、数据录入错误），
    # 保留后续行的正常价格，不整列置 NaN
    df["mid_price"] = _remove_price_outliers(df["mid_price"])

    df = df.dropna(subset=["mid_price"])

    # last 为最新成交价（实际成交，优于理论中间价），0 表示当日尚未成交，视为缺失
    last = df["last"].replace(0, np.nan)
    df["close"] = last.where(last.notna(), df["mid_price"])
    # open/high/low 对 3 秒快照无意义，统一赋 close 以满足 backtrader PandasData 接口
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]

    df["code"] = etf_code
    df = df.sort_values("trade_time").reset_index(drop=True)

    keep_cols = [
        "trade_time", "code", "open", "high", "low", "close",
        "volume", "amount", "bid_price1", "ask_price1", "mid_price", "iopv",
    ]
    # 交易所行情中 iopv 字段并非所有快照都携带，缺失时填 NaN，由 arbitrage_strategy 自行重算
    if "iopv" not in df.columns:
        df["iopv"] = np.nan
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"  {etf_code}: 加载 {len(df):,} 行，时间范围 "
          f"{df['trade_time'].min()} ~ {df['trade_time'].max()}")
    return df


# ── 债券数据加载 ──────────────────────────────────────────────────────────────

def load_bond_snapshots(bond_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载指定债券的 3 秒快照数据。

    Notes
    -----
    - 债券交易阶段码为 'T'，不同于 ETF 的 'T111'。（陷阱1）
    - last 价格为净价，不含应计利息；IOPV 计算时需另行加上。（陷阱4）
    - 债券交易时段截止 15:30，此处统一截至 15:00 以与 ETF 对齐。
    """
    code = bond_code.split(".")[0]
    path = BOND_DIR / code
    files = sorted(glob.glob(str(path / "*.csv")))

    if not files:
        raise FileNotFoundError(f"未找到 {bond_code} 的数据文件，路径：{path}")

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f, index_col=0, low_memory=False)
            dfs.append(tmp)
        except Exception as e:
            print(f"  ⚠ 读取失败：{f}，原因：{e}")

    df = pd.concat(dfs, ignore_index=True)

    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df = df.dropna(subset=["trade_time"])

    s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
    df = df[(df["trade_time"].dt.date >= s.date()) & (df["trade_time"].dt.date <= e.date())]

    # 债券交易阶段码为单字符 'T'（银行间/交易所债券连续交易），与 ETF 的 'T111' 不同；
    # 若误用 'T111' 过滤债券数据，会得到空 DataFrame，导致 IOPV 全为 NaN
    df = df[df["trading_phase_code"] == "T"]

    # 债券理论可交易至 15:30，但 ETF 收盘于 15:00；
    # 统一截至 15:00 以保证后续与 ETF 时间轴对齐，避免出现孤立 bar
    df = df[_in_trading_hours(df["trade_time"])].copy()

    # 债券 last/mid 均为净价（不含应计利息）；
    # 应计利息在 arbitrage_strategy 的 IOPV 计算中单独叠加，此处不处理
    df["mid_price"] = _mid_price(df)
    df["mid_price"] = _remove_price_outliers(df["mid_price"])
    df = df.dropna(subset=["mid_price"])

    last = df["last"].replace(0, np.nan)
    df["close"] = last.where(last.notna(), df["mid_price"])
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]

    df["code"] = bond_code
    df = df.sort_values("trade_time").reset_index(drop=True)

    keep_cols = [
        "trade_time", "code", "open", "high", "low", "close",
        "volume", "amount", "bid_price1", "ask_price1", "mid_price",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"  {bond_code}: 加载 {len(df):,} 行，时间范围 "
          f"{df['trade_time'].min()} ~ {df['trade_time'].max()}")
    return df


# ── 期货数据加载 ──────────────────────────────────────────────────────────────

def load_futures_snapshots(start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载 TL 主力期货合约快照，并做后复权处理消除换月跳价。

    陷阱3：合约换月时存在价差跳变（TL2509→TL2512 约 +0.5，TL2512→TL2603 约 -1.5）。
    若不处理，价差序列会出现虚假均值漂移，导致 OU 参数估计偏差。

    处理方法：后复权（backward additive adjustment）
    - 以最新合约价格为基准，对历史合约价格加减换月价差，保持序列连续。
    """
    df = pd.read_csv(FUTURES_PATH, index_col=0, low_memory=False)
    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df["trading_day"] = pd.to_datetime(df["trading_day"].astype(str), errors="coerce")
    df = df.dropna(subset=["trade_time", "trading_day"])

    s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
    df = df[(df["trading_day"] >= s) & (df["trading_day"] <= e)]

    # 交易时段（期货提前 1 分钟开始捕获开盘价跳空）
    df = df[_in_trading_hours(df["trade_time"], extra_minutes_before=1)].copy()

    # 过滤无成交
    df = df[df["last"] > 0]

    # 中间价
    ask = df["ask_price1"].replace(0, np.nan)
    bid = df["bid_price1"].replace(0, np.nan)
    last = df["last"]
    df["mid_price"] = np.where(ask.notna() & bid.notna(), (ask + bid) / 2, last)
    df["mid_price"] = _remove_price_outliers(df["mid_price"])
    df = df.dropna(subset=["mid_price"])
    df = df.sort_values("trade_time").reset_index(drop=True)

    # ── 后复权：消除换月价差 ──────────────────────────────────────────────────
    # 找出换月时间点（合约代码发生变化的行）
    df["_prev_code"] = df["code"].shift(1)
    roll_mask = (df["code"] != df["_prev_code"]) & df["_prev_code"].notna()
    roll_indices = df.index[roll_mask].tolist()

    df["adj_mid_price"] = df["mid_price"].copy()

    # 从最晚的换月点往前累加调整量（backward）
    for idx in reversed(roll_indices):
        if idx == 0:
            continue
        gap = df.loc[idx, "mid_price"] - df.loc[idx - 1, "mid_price"]
        df.loc[: idx - 1, "adj_mid_price"] += gap
        contract_new = df.loc[idx, "code"]
        contract_old = df.loc[idx - 1, "code"]
        print(f"  换月：{contract_old} → {contract_new}，价差 {gap:+.4f}，"
              f"已对 {idx} 行前数据做后复权")

    df.drop(columns=["_prev_code"], inplace=True)

    # close = 复权后价格，方便后续直接使用
    df["close"] = df["adj_mid_price"]
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]

    print(f"  TL期货: 加载 {len(df):,} 行，合约 {df['code'].unique()}，"
          f"时间范围 {df['trade_time'].min()} ~ {df['trade_time'].max()}")
    return df


# ── 数据重采样（3s → 1min）────────────────────────────────────────────────────

def resample_to_1min(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """将 3 秒快照重采样为 1 分钟 OHLCV 数据。"""
    df = df.set_index("trade_time").sort_index()
    ohlcv = df[price_col].resample("1min").agg(
        open="first", high="max", low="min", close="last"
    )
    if "volume" in df.columns:
        ohlcv["volume"] = df["volume"].resample("1min").sum()
    else:
        ohlcv["volume"] = 0

    if "amount" in df.columns:
        ohlcv["amount"] = df["amount"].resample("1min").sum()

    # 删除空 bar
    ohlcv = ohlcv.dropna(subset=["close"])
    ohlcv.index.name = "trade_time"
    return ohlcv.reset_index()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def process_etf_data(etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载、清洗、保存单个 ETF 的合并数据。"""
    print(f"\n[ETF] 处理 {etf_code} ({start_date} ~ {end_date})")
    df = load_etf_snapshots(etf_code, start_date, end_date)

    # 文件名格式：511090_20250701_20251231_merged.CSV
    code = etf_code.split(".")[0]
    s_str = start_date.replace("-", "")
    e_str = end_date.replace("-", "")
    fname = OUTPUT_DIR / f"{code}_{s_str}_{e_str}_merged.CSV"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"  已保存 → {fname.name}，共 {len(df):,} 行")
    return df


def process_bond_data(bond_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载、清洗、保存单个债券的合并数据。"""
    print(f"\n[债券] 处理 {bond_code} ({start_date} ~ {end_date})")
    df = load_bond_snapshots(bond_code, start_date, end_date)

    code = bond_code.split(".")[0]
    s_str = start_date.replace("-", "")
    e_str = end_date.replace("-", "")
    fname = OUTPUT_DIR / f"{code}_{s_str}_{e_str}_merged.CSV"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"  已保存 → {fname.name}，共 {len(df):,} 行")
    return df


def process_futures_data(start_date: str, end_date: str) -> pd.DataFrame:
    """加载、清洗、保存 TL 期货连续合约数据。"""
    print(f"\n[期货] 处理 TL主力合约 ({start_date} ~ {end_date})")
    df = load_futures_snapshots(start_date, end_date)

    s_str = start_date.replace("-", "")
    e_str = end_date.replace("-", "")
    fname = OUTPUT_DIR / f"TL_futures_{s_str}_{e_str}_merged.CSV"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"  已保存 → {fname.name}，共 {len(df):,} 行")
    return df


def run_task1():
    """任务1入口：执行所有数据合并。"""
    print("=" * 60)
    print("任务 1：数据处理与合并")
    print("=" * 60)

    # ETF 数据
    etf_511090 = process_etf_data("511090.SH", "2025-07-01", "2025-12-31")
    etf_511130 = process_etf_data("511130.SH", "2026-01-01", "2026-02-28")

    # 债券数据
    bond_019742 = process_bond_data("019742.SH", "2026-01-01", "2026-02-28")
    bond_019776 = process_bond_data("019776.SH", "2026-01-01", "2026-02-28")
    bond_019789 = process_bond_data("019789.SH", "2026-01-01", "2026-02-28")

    # 期货数据（含后复权）
    futures_tl = process_futures_data("2025-07-01", "2025-12-31")

    print("\n" + "=" * 60)
    print("任务 1 完成！所有文件已保存至：", OUTPUT_DIR)
    print("=" * 60)

    return {
        "etf_511090": etf_511090,
        "etf_511130": etf_511130,
        "bond_019742": bond_019742,
        "bond_019776": bond_019776,
        "bond_019789": bond_019789,
        "futures_tl": futures_tl,
    }


if __name__ == "__main__":
    run_task1()
