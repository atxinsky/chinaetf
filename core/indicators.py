# coding=utf-8
"""
技术指标计算模块
支持EMA, ADX, ATR, 动量等常用指标
"""

import pandas as pd
import numpy as np
from typing import Optional


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    计算指数移动平均线

    Args:
        series: 价格序列
        period: 周期

    Returns:
        EMA序列
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    计算简单移动平均线

    Args:
        series: 价格序列
        period: 周期

    Returns:
        SMA序列
    """
    return series.rolling(window=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均真实波幅 (Average True Range)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期

    Returns:
        ATR序列
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均趋向指数 (Average Directional Index)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期

    Returns:
        ADX序列
    """
    # 计算+DM和-DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # 计算TR
    tr = atr(high, low, close, 1)  # 使用周期1得到TR

    # 平滑
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)

    # 计算DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    # 计算ADX
    adx_value = dx.ewm(span=period, adjust=False).mean()

    return adx_value


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算相对强弱指数 (Relative Strength Index)

    Args:
        close: 收盘价序列
        period: 周期

    Returns:
        RSI序列
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    计算MACD指标

    Args:
        close: 收盘价序列
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期

    Returns:
        (MACD线, 信号线, 柱状图)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    计算布林带

    Args:
        close: 收盘价序列
        period: 周期
        std_dev: 标准差倍数

    Returns:
        (上轨, 中轨, 下轨)
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return upper, middle, lower


def highest(series: pd.Series, period: int) -> pd.Series:
    """
    计算N周期最高值

    Args:
        series: 价格序列
        period: 周期

    Returns:
        最高值序列
    """
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """
    计算N周期最低值

    Args:
        series: 价格序列
        period: 周期

    Returns:
        最低值序列
    """
    return series.rolling(window=period).min()


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    计算动量

    Args:
        close: 收盘价序列
        period: 周期

    Returns:
        动量序列
    """
    return close - close.shift(period)


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """
    计算变化率 (ROC)

    Args:
        close: 收盘价序列
        period: 周期

    Returns:
        ROC序列 (百分比)
    """
    return (close - close.shift(period)) / close.shift(period) * 100


def volume_ratio(volume: pd.Series, period: int = 5) -> pd.Series:
    """
    计算量比

    Args:
        volume: 成交量序列
        period: 周期

    Returns:
        量比序列
    """
    avg_volume = volume.rolling(window=period).mean()
    return volume / (avg_volume + 1e-10)


def calculate_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    计算所有技术指标

    Args:
        df: 包含OHLCV数据的DataFrame
        params: 指标参数字典

    Returns:
        带有技术指标的DataFrame
    """
    if params is None:
        from config import INDICATOR_PARAMS
        params = INDICATOR_PARAMS

    df = df.copy()

    # EMA指标
    ema_fast_period = params.get("ema_fast", 20)
    ema_slow_period = params.get("ema_slow", 60)

    df["ema_fast"] = ema(df["close"], ema_fast_period)
    df["ema_slow"] = ema(df["close"], ema_slow_period)

    # ADX
    adx_period = params.get("adx_period", 14)
    df["adx"] = adx(df["high"], df["low"], df["close"], adx_period)

    # ATR
    atr_period = params.get("atr_period", 14)
    df["atr"] = atr(df["high"], df["low"], df["close"], atr_period)

    # 高低点
    high_period = params.get("high_period", 20)
    df["high_20"] = highest(df["high"], high_period).shift(1)  # 前N日最高
    df["low_20"] = lowest(df["low"], high_period).shift(1)  # 前N日最低

    # RSI
    df["rsi"] = rsi(df["close"], 14)

    # MACD
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # 量比
    if "volume" in df.columns:
        df["volume_ratio"] = volume_ratio(df["volume"], 5)

    # 金叉死叉信号
    df["golden_cross"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df["death_cross"] = (df["ema_fast"] < df["ema_slow"]).astype(int)

    # 趋势强度
    df["trend_strength"] = df["adx"]

    # 波动率 (ATR/Close)
    df["volatility"] = df["atr"] / df["close"]

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算收益率相关指标

    Args:
        df: 包含收盘价的DataFrame

    Returns:
        带有收益率指标的DataFrame
    """
    df = df.copy()

    # 日收益率
    df["daily_return"] = df["close"].pct_change()

    # 累计收益率
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

    # N日收益率
    for n in [5, 10, 20, 60]:
        df[f"return_{n}d"] = df["close"].pct_change(n)

    return df
