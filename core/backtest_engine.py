# coding=utf-8
"""
ETF回测引擎
支持策略回测、绩效计算、交易记录等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """持仓信息"""
    code: str
    shares: int = 0
    avg_price: float = 0.0
    entry_date: str = ""
    entry_price: float = 0.0
    highest_price: float = 0.0
    stop_price: float = 0.0
    market_value: float = 0.0


@dataclass
class Trade:
    """交易记录"""
    date: str
    code: str
    direction: str  # "BUY" or "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float

    # 收益指标
    total_return: float  # 总收益率
    annual_return: float  # 年化收益率
    benchmark_return: float  # 基准收益率
    excess_return: float  # 超额收益率

    # 风险指标
    max_drawdown: float  # 最大回撤
    max_drawdown_duration: int  # 最大回撤持续天数
    volatility: float  # 波动率
    downside_volatility: float  # 下行波动率

    # 风险调整收益
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # 卡玛比率

    # 交易统计
    total_trades: int  # 总交易次数
    win_trades: int  # 盈利次数
    lose_trades: int  # 亏损次数
    win_rate: float  # 胜率
    profit_loss_ratio: float  # 盈亏比
    avg_win: float  # 平均盈利
    avg_loss: float  # 平均亏损
    max_win: float  # 最大单笔盈利
    max_loss: float  # 最大单笔亏损
    avg_holding_days: float  # 平均持仓天数

    # 详细数据
    equity_curve: pd.DataFrame = None  # 权益曲线
    trades: List[Trade] = None  # 交易记录
    positions_history: pd.DataFrame = None  # 持仓历史


class BacktestContext:
    """回测上下文，供策略使用"""

    def __init__(self, engine: 'BacktestEngine'):
        self._engine = engine

        # 策略参数（可在initialize中设置）
        self.base_position = 0.18
        self.atr_multiplier = 2.5
        self.max_loss = 0.07
        self.trail_start = 0.15
        self.trail_stop = 0.06
        self.max_hold = 120
        self.cooldown = 3

        # 内部状态
        self.cooldown_dict = {}
        self.entry_prices = {}
        self.entry_dates = {}
        self.highest = {}
        self.stops = {}

        # 特殊标的分类
        self.overseas = ['513100.SH', '159941.SZ', '518880.SH']
        self.high_vol = ['588000.SH', '516010.SH', '512480.SH']

    @property
    def current_date(self) -> str:
        return self._engine._current_date

    @property
    def current_dt(self) -> datetime:
        return datetime.strptime(self._engine._current_date, "%Y-%m-%d")

    @property
    def data(self) -> pd.DataFrame:
        return self._engine._current_data

    def get_account_positions(self) -> Dict[str, Position]:
        return self._engine._positions.copy()

    def get_position(self, code: str) -> Optional[Position]:
        return self._engine._positions.get(code)

    def get_cash(self) -> float:
        return self._engine._cash

    def get_total_value(self) -> float:
        return self._engine._get_total_value()

    def order_target_percent(self, code: str, target_percent: float):
        """按目标百分比下单"""
        self._engine._order_target_percent(code, target_percent)

    def order_target_value(self, code: str, target_value: float):
        """按目标金额下单"""
        self._engine._order_target_value(code, target_value)

    def order_shares(self, code: str, shares: int):
        """按股数下单"""
        self._engine._order_shares(code, shares)


class BacktestEngine:
    """
    ETF回测引擎

    功能：
    1. 事件驱动回测
    2. 支持多标的组合
    3. 计算完整绩效指标
    4. 生成交易记录和权益曲线
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0001,
        slippage: float = 0.0001,
        benchmark: str = "510300.SH"
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage: 滑点
            benchmark: 基准代码
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.benchmark = benchmark

        # 回测状态
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._equity_history: List[dict] = []
        self._positions_history: List[dict] = []

        # 当前回测状态
        self._current_date = ""
        self._current_data = None
        self._all_data: Dict[str, pd.DataFrame] = {}
        self._benchmark_data: pd.DataFrame = None

        # 策略回调
        self._initialize_func = None
        self._handle_data_func = None
        self._context = None

    def set_strategy(
        self,
        initialize: Callable = None,
        handle_data: Callable = None
    ):
        """
        设置策略函数

        Args:
            initialize: 初始化函数
            handle_data: 每日处理函数
        """
        self._initialize_func = initialize
        self._handle_data_func = handle_data

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        benchmark_data: pd.DataFrame = None
    ) -> BacktestResult:
        """
        运行回测

        Args:
            data: {code: DataFrame} 数据字典
            start_date: 开始日期
            end_date: 结束日期
            benchmark_data: 基准数据

        Returns:
            BacktestResult
        """
        logger.info(f"开始回测: {start_date} ~ {end_date}")

        # 初始化
        self._reset()
        self._all_data = data
        self._benchmark_data = benchmark_data

        # 创建上下文
        self._context = BacktestContext(self)

        # 调用策略初始化
        if self._initialize_func:
            self._initialize_func(self._context)

        # 获取所有交易日期
        all_dates = set()
        for code, df in data.items():
            if "date" in df.columns:
                all_dates.update(df["date"].tolist())

        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        if not trading_dates:
            raise ValueError("没有可用的交易日期")

        logger.info(f"交易日期: {len(trading_dates)}天")

        # 遍历每个交易日
        for date in trading_dates:
            self._current_date = date

            # 获取当日数据
            self._current_data = self._get_daily_data(date)

            # 更新持仓市值
            self._update_positions_market_value()

            # 调用策略处理函数
            if self._handle_data_func and len(self._current_data) > 0:
                self._handle_data_func(self._context, None)

            # 记录权益
            self._record_equity()

        # 计算回测结果
        result = self._calculate_result(start_date, end_date)

        logger.info(f"回测完成: 总收益 {result.total_return*100:.2f}%")

        return result

    def _reset(self):
        """重置回测状态"""
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._equity_history = []
        self._positions_history = []

    def _get_daily_data(self, date: str) -> pd.DataFrame:
        """获取指定日期的所有数据"""
        rows = []
        for code, df in self._all_data.items():
            day_df = df[df["date"] == date]
            if len(day_df) > 0:
                row = day_df.iloc[0].to_dict()
                row["instrument"] = code
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def _update_positions_market_value(self):
        """更新持仓市值"""
        for code, pos in self._positions.items():
            if code in self._all_data:
                df = self._all_data[code]
                day_df = df[df["date"] == self._current_date]
                if len(day_df) > 0:
                    price = day_df.iloc[0]["close"]
                    pos.market_value = pos.shares * price

                    # 更新最高价
                    if price > pos.highest_price:
                        pos.highest_price = price

    def _get_total_value(self) -> float:
        """获取账户总价值"""
        total = self._cash
        for pos in self._positions.values():
            total += pos.market_value
        return total

    def _get_price(self, code: str) -> Optional[float]:
        """获取当前价格"""
        if self._current_data is None or len(self._current_data) == 0:
            return None

        row = self._current_data[self._current_data["instrument"] == code]
        if len(row) > 0:
            return row.iloc[0]["close"]
        return None

    def _order_target_percent(self, code: str, target_percent: float):
        """按目标百分比下单"""
        total_value = self._get_total_value()
        target_value = total_value * target_percent
        self._order_target_value(code, target_value)

    def _order_target_value(self, code: str, target_value: float):
        """按目标金额下单"""
        price = self._get_price(code)
        if price is None:
            logger.warning(f"无法获取价格: {code}")
            return

        current_value = 0
        if code in self._positions:
            current_value = self._positions[code].market_value

        diff_value = target_value - current_value

        if abs(diff_value) < 100:  # 忽略小额调整
            return

        # 计算股数（ETF一手100股）
        shares = int(diff_value / price / 100) * 100

        if shares != 0:
            self._order_shares(code, shares)

    def _order_shares(self, code: str, shares: int):
        """按股数下单"""
        price = self._get_price(code)
        if price is None:
            logger.warning(f"无法获取价格: {code}")
            return

        # 计算滑点后价格
        if shares > 0:
            exec_price = price * (1 + self.slippage)
        else:
            exec_price = price * (1 - self.slippage)

        amount = abs(shares) * exec_price
        commission = amount * self.commission_rate
        commission = max(commission, 0.1)  # 最低手续费

        # 买入
        if shares > 0:
            total_cost = amount + commission
            if total_cost > self._cash:
                # 资金不足，调整股数
                available = self._cash - commission
                shares = int(available / exec_price / 100) * 100
                if shares <= 0:
                    return
                amount = shares * exec_price
                commission = max(amount * self.commission_rate, 0.1)
                total_cost = amount + commission

            self._cash -= total_cost

            if code not in self._positions:
                self._positions[code] = Position(
                    code=code,
                    shares=shares,
                    avg_price=exec_price,
                    entry_date=self._current_date,
                    entry_price=exec_price,
                    highest_price=exec_price,
                    market_value=shares * exec_price
                )
            else:
                pos = self._positions[code]
                total_shares = pos.shares + shares
                pos.avg_price = (pos.shares * pos.avg_price + shares * exec_price) / total_shares
                pos.shares = total_shares
                pos.market_value = total_shares * exec_price

            trade = Trade(
                date=self._current_date,
                code=code,
                direction="BUY",
                price=exec_price,
                shares=shares,
                amount=amount,
                commission=commission
            )
            self._trades.append(trade)

            logger.debug(f"买入 {code}: {shares}股 @ {exec_price:.3f}")

        # 卖出
        elif shares < 0:
            shares = abs(shares)

            if code not in self._positions:
                return

            pos = self._positions[code]
            shares = min(shares, pos.shares)

            if shares <= 0:
                return

            amount = shares * exec_price
            commission = max(amount * self.commission_rate, 0.1)

            self._cash += amount - commission

            # 计算盈亏
            pnl = (exec_price - pos.avg_price) * shares - commission
            pnl_pct = (exec_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0

            trade = Trade(
                date=self._current_date,
                code=code,
                direction="SELL",
                price=exec_price,
                shares=shares,
                amount=amount,
                commission=commission,
                pnl=pnl,
                pnl_pct=pnl_pct
            )
            self._trades.append(trade)

            pos.shares -= shares
            pos.market_value = pos.shares * exec_price

            if pos.shares <= 0:
                del self._positions[code]

            logger.debug(f"卖出 {code}: {shares}股 @ {exec_price:.3f}, 盈亏: {pnl:.2f}")

    def _record_equity(self):
        """记录权益"""
        total_value = self._get_total_value()

        record = {
            "date": self._current_date,
            "cash": self._cash,
            "market_value": sum(p.market_value for p in self._positions.values()),
            "total_value": total_value,
            "positions_count": len(self._positions)
        }

        self._equity_history.append(record)

        # 记录持仓详情
        for code, pos in self._positions.items():
            self._positions_history.append({
                "date": self._current_date,
                "code": code,
                "shares": pos.shares,
                "market_value": pos.market_value,
                "entry_price": pos.entry_price,
                "entry_date": pos.entry_date
            })

    def _calculate_result(self, start_date: str, end_date: str) -> BacktestResult:
        """计算回测结果"""
        # 构建权益曲线
        equity_df = pd.DataFrame(self._equity_history)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")
        equity_df["return"] = equity_df["total_value"].pct_change()
        equity_df["cumulative_return"] = equity_df["total_value"] / self.initial_capital - 1

        # 基本指标
        final_value = equity_df["total_value"].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 计算年化收益率
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 计算基准收益率
        benchmark_return = 0
        if self._benchmark_data is not None and len(self._benchmark_data) > 0:
            bench_df = self._benchmark_data[
                (self._benchmark_data["date"] >= start_date) &
                (self._benchmark_data["date"] <= end_date)
            ]
            if len(bench_df) > 1:
                benchmark_return = bench_df.iloc[-1]["close"] / bench_df.iloc[0]["close"] - 1

        excess_return = total_return - benchmark_return

        # 计算最大回撤
        rolling_max = equity_df["total_value"].cummax()
        drawdown = (equity_df["total_value"] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # 计算最大回撤持续时间
        drawdown_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0

        # 波动率
        volatility = equity_df["return"].std() * np.sqrt(252)

        # 下行波动率
        negative_returns = equity_df["return"][equity_df["return"] < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0

        # 夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        excess_return_daily = equity_df["return"].mean() - risk_free_rate / 252
        sharpe_ratio = excess_return_daily / equity_df["return"].std() * np.sqrt(252) if equity_df["return"].std() > 0 else 0

        # 索提诺比率
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0

        # 卡玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 交易统计
        sell_trades = [t for t in self._trades if t.direction == "SELL"]
        total_trades = len(sell_trades)

        win_trades = len([t for t in sell_trades if t.pnl > 0])
        lose_trades = len([t for t in sell_trades if t.pnl < 0])

        win_rate = win_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in sell_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in sell_trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        max_win = max(wins) if wins else 0
        max_loss = max(losses) if losses else 0

        # 平均持仓天数
        avg_holding_days = 0
        if sell_trades:
            holding_days = []
            for trade in sell_trades:
                # 查找对应的买入交易
                buy_trades = [t for t in self._trades
                              if t.code == trade.code and t.direction == "BUY" and t.date <= trade.date]
                if buy_trades:
                    entry_date = datetime.strptime(buy_trades[-1].date, "%Y-%m-%d")
                    exit_date = datetime.strptime(trade.date, "%Y-%m-%d")
                    holding_days.append((exit_date - entry_date).days)
            avg_holding_days = np.mean(holding_days) if holding_days else 0

        # 持仓历史
        positions_df = pd.DataFrame(self._positions_history) if self._positions_history else None

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            volatility=volatility,
            downside_volatility=downside_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            win_trades=win_trades,
            lose_trades=lose_trades,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            avg_holding_days=avg_holding_days,
            equity_curve=equity_df,
            trades=self._trades,
            positions_history=positions_df
        )
