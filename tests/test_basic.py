# coding=utf-8
"""
基础功能测试
运行: python -m pytest tests/test_basic.py -v
或者: python tests/test_basic.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def test_imports():
    """测试模块导入"""
    print("测试模块导入...")

    # config
    import config
    assert hasattr(config, 'ETF_POOLS')
    assert hasattr(config, 'ALL_ETFS')
    assert hasattr(config, 'DATABASE_PATH')
    print("  [OK] config")

    # indicators
    from core.indicators import ema, adx, atr, calculate_indicators
    print("  [OK] indicators")

    # data_service
    from core.data_service import ETFDataService, get_data_service
    print("  [OK] data_service")

    # backtest_engine
    from core.backtest_engine import BacktestEngine, BacktestContext, BacktestResult
    print("  [OK] backtest_engine")

    # strategies
    from strategies.base import BaseStrategy
    from strategies.bigbrother_v14 import BigBrotherV14
    print("  [OK] strategies")

    print("模块导入测试通过!")


def test_indicators():
    """测试技术指标计算"""
    print("\n测试技术指标...")

    import pandas as pd
    import numpy as np
    from core.indicators import ema, adx, atr, calculate_indicators

    # 创建测试数据
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': close + np.random.randn(n) * 0.5,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n)
    })

    # 测试EMA
    ema_20 = ema(df['close'], 20)
    assert len(ema_20) == n
    assert not ema_20.isna().all()
    print("  [OK] EMA")

    # 测试ATR
    atr_14 = atr(df['high'], df['low'], df['close'], 14)
    assert len(atr_14) == n
    assert (atr_14 >= 0).all()
    print("  [OK] ATR")

    # 测试ADX
    adx_14 = adx(df['high'], df['low'], df['close'], 14)
    assert len(adx_14) == n
    print("  [OK] ADX")

    # 测试calculate_indicators
    df_with_indicators = calculate_indicators(df)
    assert 'ema_fast' in df_with_indicators.columns
    assert 'ema_slow' in df_with_indicators.columns
    assert 'adx' in df_with_indicators.columns
    assert 'atr' in df_with_indicators.columns
    print("  [OK] calculate_indicators")

    print("技术指标测试通过!")


def test_backtest_engine():
    """测试回测引擎"""
    print("\n测试回测引擎...")

    import pandas as pd
    import numpy as np
    from core.backtest_engine import BacktestEngine, BacktestContext
    from core.indicators import calculate_indicators

    # 创建测试数据
    np.random.seed(42)
    n = 200

    def create_test_data(code, trend=0.001):
        dates = pd.date_range('2024-01-01', periods=n)
        close = 10 + np.cumsum(np.random.randn(n) * 0.1 + trend)
        close = np.maximum(close, 1)  # 确保价格为正
        high = close + np.random.rand(n) * 0.2
        low = close - np.random.rand(n) * 0.2

        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': close + np.random.randn(n) * 0.05,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        return calculate_indicators(df)

    data = {
        '513100.SH': create_test_data('513100.SH', 0.002),
        '518880.SH': create_test_data('518880.SH', 0.001),
        '510300.SH': create_test_data('510300.SH', 0.001),
    }

    # 简单策略
    def initialize(context):
        context.position_size = 0.3

    def handle_data(context, data):
        today_df = context.data
        if today_df is None or len(today_df) == 0:
            return

        positions = context.get_account_positions()

        for _, row in today_df.iterrows():
            code = row['instrument']
            if code == '510300.SH':
                continue

            # 简单的金叉买入策略
            if pd.notna(row.get('ema_fast')) and pd.notna(row.get('ema_slow')):
                golden = row['ema_fast'] > row['ema_slow']

                if code not in positions and golden:
                    context.order_target_percent(code, context.position_size)
                elif code in positions and not golden:
                    context.order_target_percent(code, 0)

    # 运行回测
    engine = BacktestEngine(
        initial_capital=1000000,
        commission_rate=0.0001,
        slippage=0.0001,
        benchmark='510300.SH'
    )

    engine.set_strategy(initialize, handle_data)

    result = engine.run(
        data=data,
        start_date='2024-03-01',
        end_date='2024-07-01',
        benchmark_data=data['510300.SH']
    )

    assert result is not None
    assert result.total_trades >= 0
    assert result.equity_curve is not None
    print(f"  回测结果: 收益率={result.total_return*100:.2f}%, 交易次数={result.total_trades}")
    print("  [OK] BacktestEngine")

    print("回测引擎测试通过!")


def test_strategy():
    """测试策略类"""
    print("\n测试策略...")

    from strategies.bigbrother_v14 import BigBrotherV14

    # 创建策略实例
    strategy = BigBrotherV14(
        base_position=0.2,
        max_loss=0.08
    )

    assert strategy.name == "BigBrother_V14"
    assert strategy.params['base_position'] == 0.2
    assert strategy.params['max_loss'] == 0.08
    assert len(strategy.pool) == 8
    print("  [OK] BigBrotherV14")

    print("策略测试通过!")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("ETF量化系统 - 基础功能测试")
    print("=" * 50)

    try:
        test_imports()
        test_indicators()
        test_backtest_engine()
        test_strategy()

        print("\n" + "=" * 50)
        print("所有测试通过!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
