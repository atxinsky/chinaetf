# coding=utf-8
"""
ETF量化系统启动入口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from config import STREAMLIT_PORT


def run_web():
    """启动Streamlit Web界面"""
    import subprocess

    app_path = ROOT_DIR / "app" / "main.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]

    print(f"启动ETF量化系统...")
    print(f"访问地址: http://localhost:{STREAMLIT_PORT}")

    subprocess.run(cmd)


def update_data():
    """更新数据"""
    from core.data_service import get_data_service
    from config import ALL_ETFS

    print("开始更新ETF数据...")

    ds = get_data_service()
    results = ds.update_all(force=False)

    for code, rows in results.items():
        name = ALL_ETFS.get(code, "")
        if rows >= 0:
            print(f"  {code} {name}: +{rows}条")
        else:
            print(f"  {code} {name}: 更新失败")

    print("数据更新完成!")


def run_backtest():
    """运行回测（命令行模式）"""
    from core.data_service import get_data_service
    from core.backtest_engine import BacktestEngine
    from strategies.bigbrother_v14 import BigBrotherV14

    print("运行BigBrother V14回测...")

    # 默认参数
    start_date = "2021-01-01"
    end_date = "2026-01-08"
    initial_capital = 1000000

    # 加载数据
    ds = get_data_service()

    pool = BigBrotherV14.DEFAULT_POOL + ["000300.SH", "510300.SH"]

    print(f"加载数据: {len(pool)}个标的...")
    data = {}
    for code in pool:
        df = ds.get_data_with_indicators(code, start_date, end_date)
        if len(df) > 0:
            data[code] = df
            print(f"  {code}: {len(df)}条")
        else:
            print(f"  {code}: 无数据")

    if not data:
        print("错误: 无可用数据，请先运行 update_data")
        return

    # 创建策略
    strategy = BigBrotherV14()

    # 创建引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.0001,
        slippage=0.0001,
        benchmark="510300.SH"
    )

    engine.set_strategy(
        initialize=strategy.initialize,
        handle_data=strategy.handle_data
    )

    # 运行回测
    result = engine.run(
        data=data,
        start_date=start_date,
        end_date=end_date,
        benchmark_data=data.get("510300.SH")
    )

    # 输出结果
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"回测区间: {result.start_date} ~ {result.end_date}")
    print(f"初始资金: ¥{result.initial_capital:,.0f}")
    print(f"最终价值: ¥{result.final_value:,.0f}")
    print("-"*60)
    print(f"累计收益: {result.total_return*100:.2f}%")
    print(f"年化收益: {result.annual_return*100:.2f}%")
    print(f"基准收益: {result.benchmark_return*100:.2f}%")
    print(f"超额收益: {result.excess_return*100:.2f}%")
    print("-"*60)
    print(f"最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"波动率: {result.volatility*100:.2f}%")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"卡玛比率: {result.calmar_ratio:.2f}")
    print("-"*60)
    print(f"总交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate*100:.1f}%")
    print(f"盈亏比: {result.profit_loss_ratio:.2f}")
    print(f"平均持仓: {result.avg_holding_days:.1f}天")
    print("="*60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ETF量化系统")
    parser.add_argument("command", choices=["web", "update", "backtest"],
                        help="命令: web=启动界面, update=更新数据, backtest=运行回测")

    args = parser.parse_args()

    if args.command == "web":
        run_web()
    elif args.command == "update":
        update_data()
    elif args.command == "backtest":
        run_backtest()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 默认启动web
        run_web()
    else:
        main()
