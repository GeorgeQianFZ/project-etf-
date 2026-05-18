"""
主回测脚本
执行顺序：
  1. 任务1：数据处理与合并（data_processor.py）
  2. 子任务 2a：511130 ETF 申赎套利（arbitrage_strategy.py）
  3. 子任务 2b：511090 vs TL 期货 OU 配对交易（pairs_trading_strategy.py）
  4. 汇总所有绩效指标，输出至 performance_summary.csv
"""

import time
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("/Users/bulusiweien/Downloads/钱方舟测试结果")


def main():
    total_start = time.time()
    print("\n" + "=" * 70)
    print("  汇远盈量化研究岗 机考测试 202603")
    print("=" * 70)

    # ── 任务 1：数据处理 ──────────────────────────────────────────────────────
    print("\n>>> 任务 1：数据处理与合并")
    from data_processor import run_task1
    run_task1()

    # ── 子任务 2a：ETF 申赎套利 ───────────────────────────────────────────────
    print("\n>>> 子任务 2a：ETF 申赎套利策略（511130，2026-01 ~ 2026-02）")
    from arbitrage_strategy import run_etf_arbitrage_backtest
    result_2a = run_etf_arbitrage_backtest(
        max_capital=10_000_000,
        max_baskets_per_trade=5,
    )

    # ── 子任务 2b：OU 配对交易 ────────────────────────────────────────────────
    print("\n>>> 子任务 2b：ETF-期货 OU 配对交易（511090 vs TL，2025-07 ~ 2025-12）")
    from pairs_trading_strategy import run_pairs_trading_backtest
    result_2b = run_pairs_trading_backtest(
        entry_z=1.5,
        exit_z=0.5,
        max_futures=8,
        max_capital=10_000_000,
        update_freq=180,
    )

    # ── 子任务 3：策略优化与风险分析 ─────────────────────────────────────────
    print("\n>>> 任务 3：Optuna 参数优化 + 敏感性分析 + 风险控制回测")
    from strategy_optimizer import run_task3
    run_task3(n_optuna_trials=80)

    # ── 汇总绩效指标 ──────────────────────────────────────────────────────────
    summary_rows = []
    summary_rows.append({"策略": "2a_ETF申赎套利(511130)", **result_2a["perf"]})
    summary_rows.append({"策略": "2b_OU配对交易(511090vsTL)", **result_2b["perf"]})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "performance_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  所有任务完成！")
    print(f"  总耗时：{elapsed:.1f} 秒")
    print(f"  结果目录：{OUTPUT_DIR}")
    print("=" * 70)

    print("\n── 绩效汇总 ──")
    print(summary_df.to_string(index=False))

    # 列出所有输出文件
    print("\n── 输出文件清单 ──")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.suffix in (".csv", ".png", ".py"):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:<50} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
