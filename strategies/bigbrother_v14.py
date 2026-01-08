# coding=utf-8
"""
BigBrother V14 策略 - ETF轮动趋势跟踪

基于V13的59.35%收益，扩展标的池

特点：
1. 固定精选8个ETF池（优于动态筛选）
2. EMA金叉/死叉趋势判断
3. ADX趋势强度过滤
4. ATR动态止损 + 硬止损
5. 追踪止盈
6. A股大盘过滤（海外标的豁免）

标的池：
- 513100.SH 纳指ETF (海外科技)
- 513050.SH 中概互联 (中概股)
- 512480.SH 半导体ETF (国产替代)
- 515030.SH 新能车ETF (新能源)
- 518880.SH 黄金ETF (避险资产)
- 512890.SH 红利低波 (防守型)
- 588000.SH 科创50 (硬科技)
- 516010.SH 游戏动漫 (AI应用)
"""

from datetime import datetime, timedelta
import pandas as pd
import logging

from .base import BaseStrategy, register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class BigBrotherV14(BaseStrategy):
    """BigBrother V14 ETF轮动策略"""

    name = "BigBrother_V14"

    description = """
    BigBrother V14 - ETF趋势轮动策略

    核心逻辑：
    - 入场：EMA(20)金叉EMA(60) + ADX>20 + 价格接近20日高点
    - 出场：硬止损7% / ATR止损 / 追踪止盈(15%触发,6%回撤) / 死叉

    历史表现（2021-2026）：
    - 累计收益：~50%
    - 年化收益：~8.5%
    - 最大回撤：<10%
    - 夏普比率：~0.59
    """

    parameters = {
        "base_position": {
            "name": "基础仓位",
            "type": "float",
            "default": 0.18,
            "min": 0.05,
            "max": 0.30,
            "step": 0.01,
            "description": "每个标的的基础仓位比例"
        },
        "atr_multiplier": {
            "name": "ATR止损倍数",
            "type": "float",
            "default": 2.5,
            "min": 1.5,
            "max": 4.0,
            "step": 0.1,
            "description": "ATR止损倍数"
        },
        "max_loss": {
            "name": "硬止损比例",
            "type": "float",
            "default": 0.07,
            "min": 0.05,
            "max": 0.15,
            "step": 0.01,
            "description": "最大单笔亏损比例"
        },
        "trail_start": {
            "name": "追踪止盈触发",
            "type": "float",
            "default": 0.15,
            "min": 0.08,
            "max": 0.30,
            "step": 0.01,
            "description": "触发追踪止盈的盈利比例"
        },
        "trail_stop": {
            "name": "追踪回撤比例",
            "type": "float",
            "default": 0.06,
            "min": 0.03,
            "max": 0.15,
            "step": 0.01,
            "description": "从最高点回撤触发止盈的比例"
        },
        "max_hold": {
            "name": "最长持仓天数",
            "type": "int",
            "default": 120,
            "min": 30,
            "max": 365,
            "step": 10,
            "description": "超过该天数且收益不佳则轮换"
        },
        "cooldown": {
            "name": "冷却天数",
            "type": "int",
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
            "description": "止损后的冷却期"
        },
        "adx_threshold": {
            "name": "ADX阈值",
            "type": "int",
            "default": 20,
            "min": 15,
            "max": 30,
            "step": 1,
            "description": "ADX趋势强度阈值"
        },
    }

    # 默认ETF池
    DEFAULT_POOL = [
        "513100.SH",  # 纳指ETF
        "513050.SH",  # 中概互联
        "512480.SH",  # 半导体ETF
        "515030.SH",  # 新能车ETF
        "518880.SH",  # 黄金ETF
        "512890.SH",  # 红利低波
        "588000.SH",  # 科创50
        "516010.SH",  # 游戏动漫
    ]

    # 海外/商品标的（不受A股大盘过滤）
    OVERSEAS = ["513100.SH", "159941.SZ", "518880.SH"]

    # 高波动标的（降低仓位）
    HIGH_VOL = ["588000.SH", "516010.SH", "512480.SH"]

    # A股基准
    BENCHMARK = "000300.SH"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = kwargs.get("pool", self.DEFAULT_POOL)

    def initialize(self, context):
        """初始化策略"""
        # 设置参数到context
        context.base_position = self.params.get("base_position", 0.18)
        context.atr_multiplier = self.params.get("atr_multiplier", 2.5)
        context.max_loss = self.params.get("max_loss", 0.07)
        context.trail_start = self.params.get("trail_start", 0.15)
        context.trail_stop = self.params.get("trail_stop", 0.06)
        context.max_hold = self.params.get("max_hold", 120)
        context.cooldown = self.params.get("cooldown", 3)
        context.adx_threshold = self.params.get("adx_threshold", 20)

        # 内部状态
        context.cooldown_dict = {}
        context.entry_prices = {}
        context.entry_dates = {}
        context.highest = {}
        context.stops = {}

        # 标的分类
        context.overseas = self.OVERSEAS
        context.high_vol = self.HIGH_VOL

        logger.info(f"BigBrother V14 初始化完成, 标的池: {len(self.pool)}个")

    def handle_data(self, context, data):
        """每日处理"""
        current_date = context.current_date
        current_dt = context.current_dt

        today_df = context.data
        if today_df is None or len(today_df) == 0:
            return

        positions = context.get_account_positions()

        # --- A股大盘状态 ---
        bench_df = today_df[today_df["instrument"] == self.BENCHMARK]
        a_market_ok = True

        if len(bench_df) > 0:
            row = bench_df.iloc[0]
            ma20 = row.get("ema_fast", None)
            ma60 = row.get("ema_slow", None)
            close = row["close"]

            if ma20 and ma60 and not pd.isna(ma20) and not pd.isna(ma60):
                # 熊市: 价格<MA20<MA60
                if close < ma20 and ma20 < ma60:
                    a_market_ok = False

        # --- 遍历标的 ---
        for _, row in today_df.iterrows():
            instrument = row["instrument"]

            # 跳过基准
            if instrument == self.BENCHMARK:
                continue

            # 只处理池内标的
            if instrument not in self.pool:
                continue

            price = row["close"]

            # 检查必要字段
            fields = ["ema_fast", "ema_slow", "adx", "atr", "high_20"]
            if any(pd.isna(row.get(f)) for f in fields):
                continue

            ma20 = row["ema_fast"]
            ma60 = row["ema_slow"]
            adx = row["adx"]
            atr = row["atr"]
            high20 = row["high_20"]

            golden = ma20 > ma60
            death = ma20 < ma60

            is_overseas = instrument in context.overseas
            is_high_vol = instrument in context.high_vol

            # === 持仓处理 ===
            if instrument in positions and positions[instrument].shares > 0:
                self._handle_position(
                    context, instrument, price, ma20, ma60, adx, atr,
                    golden, death, current_date, current_dt
                )

            # === 开仓 ===
            else:
                self._handle_entry(
                    context, instrument, price, ma20, ma60, adx, atr, high20,
                    golden, is_overseas, is_high_vol, a_market_ok, current_dt
                )

    def _handle_position(self, context, instrument, price, ma20, ma60, adx, atr,
                          golden, death, current_date, current_dt):
        """处理持仓"""
        entry = context.entry_prices.get(instrument, price)
        entry_dt = context.entry_dates.get(instrument, current_dt)
        high = context.highest.get(instrument, price)
        stop = context.stops.get(instrument, 0)

        # 更新最高价
        if price > high:
            context.highest[instrument] = price
            high = price

        pnl = (price - entry) / entry
        days = (current_dt - entry_dt).days

        exit_flag = False
        reason = ""

        # 1. 硬止损
        if pnl <= -context.max_loss:
            exit_flag = True
            reason = f"止损{context.max_loss*100:.0f}%"
            context.cooldown_dict[instrument] = current_dt + timedelta(days=context.cooldown)

        # 2. ATR止损
        elif price <= stop:
            exit_flag = True
            reason = "ATR止损"
            context.cooldown_dict[instrument] = current_dt + timedelta(days=context.cooldown)

        # 3. 追踪止盈
        elif pnl >= context.trail_start:
            dd = (high - price) / high
            if dd >= context.trail_stop:
                exit_flag = True
                reason = f"止盈 | 最高:{(high-entry)/entry*100:.0f}%"

        # 4. 长期持仓检视
        elif days >= context.max_hold and pnl < 0.03:
            exit_flag = True
            reason = f"轮换{days}天"

        # 5. 死叉
        elif death:
            exit_flag = True
            reason = "死叉"

        if exit_flag:
            context.order_target_percent(instrument, 0)
            context.entry_prices.pop(instrument, None)
            context.entry_dates.pop(instrument, None)
            context.highest.pop(instrument, None)
            context.stops.pop(instrument, None)
            logger.info(f"[{current_date}] 卖出 {instrument} | {reason} | 盈亏:{pnl*100:+.1f}%")

    def _handle_entry(self, context, instrument, price, ma20, ma60, adx, atr, high20,
                       golden, is_overseas, is_high_vol, a_market_ok, current_dt):
        """处理开仓"""
        # 检查冷却期
        cd = context.cooldown_dict.get(instrument)
        if cd and current_dt < cd:
            return

        # 入场条件
        # 1. 金叉
        if not golden:
            return

        # 2. ADX > 阈值
        if adx < context.adx_threshold:
            return

        # 3. 价格接近20日高点
        if price < high20 * 0.95:
            return

        # 4. 大盘过滤（海外/商品豁免）
        if not is_overseas and not a_market_ok and adx < 30:
            return

        # 仓位计算
        vol = atr / price if price > 0 else 0.02
        adj = min(1.0, 0.02 / vol) if vol > 0 else 1.0
        pos = context.base_position * adj

        # 高波动标的降仓
        if is_high_vol:
            pos = pos * 0.7

        pos = max(0.10, min(0.25, pos))

        # 止损价
        atr_stop = price - context.atr_multiplier * atr
        hard_stop = price * (1 - context.max_loss)
        stop = max(atr_stop, hard_stop)

        context.order_target_percent(instrument, pos)
        context.entry_prices[instrument] = price
        context.entry_dates[instrument] = current_dt
        context.highest[instrument] = price
        context.stops[instrument] = stop

        tag = "[海外]" if is_overseas else ("[高波]" if is_high_vol else "")
        logger.info(f"[{context.current_date}] 买入 {instrument} {tag} | 价:{price:.3f} | ADX:{adx:.0f} | 仓位:{pos*100:.0f}%")
