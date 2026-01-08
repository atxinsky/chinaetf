# coding=utf-8
"""
回测页面 - 复刻BigQuant回测界面风格
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


def render_backtest_page():
    """渲染回测页面"""
    st.markdown("## 📈 ETF策略回测")

    # 三列布局：设置 | 参数 | 标的池
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 📅 回测设置")

        # 日期范围
        start_date = st.date_input(
            "开始日期",
            value=datetime(2021, 1, 1),
            min_value=datetime(2019, 1, 1),
            max_value=datetime.now()
        )

        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            min_value=datetime(2019, 1, 1),
            max_value=datetime.now()
        )

        # 初始资金
        initial_capital = st.number_input(
            "初始资金",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000,
            format="%d"
        )

        # 手续费
        commission = st.number_input(
            "手续费率",
            min_value=0.0,
            max_value=0.01,
            value=0.0001,
            step=0.0001,
            format="%.4f"
        )

    with col2:
        st.markdown("### ⚙️ 策略参数")

        # 策略选择
        strategy_name = st.selectbox(
            "策略选择",
            ["BigBrother V14"],
            index=0
        )

        # 策略参数
        base_position = st.slider(
            "基础仓位",
            min_value=0.05,
            max_value=0.30,
            value=0.18,
            step=0.01,
            format="%.2f"
        )

        max_loss = st.slider(
            "硬止损比例",
            min_value=0.05,
            max_value=0.15,
            value=0.07,
            step=0.01,
            format="%.2f"
        )

        atr_multiplier = st.slider(
            "ATR止损倍数",
            min_value=1.5,
            max_value=4.0,
            value=2.5,
            step=0.1,
            format="%.1f"
        )

        trail_start = st.slider(
            "追踪止盈触发",
            min_value=0.08,
            max_value=0.30,
            value=0.15,
            step=0.01,
            format="%.2f"
        )

        adx_threshold = st.slider(
            "ADX阈值",
            min_value=15,
            max_value=30,
            value=20,
            step=1
        )

    with col3:
        st.markdown("### 📋 标的池")

        from config import ETF_POOLS

        # 预设池选择
        pool_options = ["自定义"] + list(ETF_POOLS.keys()) + ["BigBrother V14 默认池"]
        selected_pool = st.selectbox("预设池", pool_options, index=len(pool_options)-1)

        # 根据选择显示ETF
        if selected_pool == "BigBrother V14 默认池":
            default_codes = [
                "513100.SH", "513050.SH", "512480.SH", "515030.SH",
                "518880.SH", "512890.SH", "588000.SH", "516010.SH"
            ]
        elif selected_pool == "自定义":
            default_codes = []
        else:
            default_codes = list(ETF_POOLS[selected_pool].keys())

        # 多选ETF
        from config import ALL_ETFS
        all_codes = list(ALL_ETFS.keys())

        selected_etfs = st.multiselect(
            "选择ETF",
            options=all_codes,
            default=default_codes,
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}"
        )

        # 添加基准
        st.markdown("**基准**")
        benchmark = st.selectbox(
            "基准指数",
            ["510300.SH (沪深300ETF)", "000300.SH (沪深300指数)"],
            label_visibility="collapsed"
        )

    st.markdown("---")

    # 运行回测按钮
    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        if not selected_etfs:
            st.error("请至少选择一个ETF标的")
            return

        run_backtest(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=initial_capital,
            commission=commission,
            strategy_name=strategy_name,
            selected_etfs=selected_etfs,
            strategy_params={
                "base_position": base_position,
                "max_loss": max_loss,
                "atr_multiplier": atr_multiplier,
                "trail_start": trail_start,
                "adx_threshold": adx_threshold
            },
            benchmark=benchmark.split(" ")[0]
        )


def run_backtest(start_date, end_date, initial_capital, commission,
                 strategy_name, selected_etfs, strategy_params, benchmark):
    """运行回测"""

    with st.spinner("正在加载数据..."):
        try:
            from core.data_service import get_data_service
            from core.backtest_engine import BacktestEngine
            from strategies.bigbrother_v14 import BigBrotherV14

            ds = get_data_service()

            # 加载数据（包括基准）
            all_codes = selected_etfs + [benchmark, "000300.SH"]
            all_codes = list(set(all_codes))

            data = {}
            progress_bar = st.progress(0)

            for i, code in enumerate(all_codes):
                # 先尝试从本地获取
                df = ds.get_data_with_indicators(code, start_date, end_date)

                if len(df) == 0:
                    st.warning(f"无数据: {code}，正在从网络获取...")
                    ds.update_data(code)
                    df = ds.get_data_with_indicators(code, start_date, end_date)

                if len(df) > 0:
                    data[code] = df

                progress_bar.progress((i + 1) / len(all_codes))

            progress_bar.empty()

            if not data:
                st.error("无法加载任何数据，请先更新数据")
                return

            st.success(f"数据加载完成: {len(data)}个标的")

        except Exception as e:
            st.error(f"数据加载失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    with st.spinner("正在运行回测..."):
        try:
            # 创建策略
            strategy = BigBrotherV14(
                pool=selected_etfs,
                **strategy_params
            )

            # 创建回测引擎
            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission_rate=commission,
                slippage=0.0001,
                benchmark=benchmark
            )

            # 设置策略
            engine.set_strategy(
                initialize=strategy.initialize,
                handle_data=strategy.handle_data
            )

            # 获取基准数据
            benchmark_data = data.get(benchmark, data.get("000300.SH"))

            # 运行回测
            result = engine.run(
                data=data,
                start_date=start_date,
                end_date=end_date,
                benchmark_data=benchmark_data
            )

            st.success("回测完成!")

            # 显示结果
            display_backtest_result(result, benchmark)

        except Exception as e:
            st.error(f"回测失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def display_backtest_result(result, benchmark):
    """显示回测结果（复刻BigQuant风格）"""

    # 核心指标卡片
    st.markdown("### 📊 绩效概览")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "累计收益",
            f"{result.total_return*100:.2f}%",
            delta=f"vs基准 {result.excess_return*100:+.2f}%"
        )

    with col2:
        st.metric(
            "年化收益",
            f"{result.annual_return*100:.2f}%"
        )

    with col3:
        st.metric(
            "最大回撤",
            f"{result.max_drawdown*100:.2f}%"
        )

    with col4:
        st.metric(
            "夏普比率",
            f"{result.sharpe_ratio:.2f}"
        )

    with col5:
        st.metric(
            "胜率",
            f"{result.win_rate*100:.1f}%"
        )

    # 第二行指标
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("基准收益", f"{result.benchmark_return*100:.2f}%")

    with col2:
        st.metric("波动率", f"{result.volatility*100:.2f}%")

    with col3:
        st.metric("卡玛比率", f"{result.calmar_ratio:.2f}")

    with col4:
        st.metric("盈亏比", f"{result.profit_loss_ratio:.2f}")

    with col5:
        st.metric("总交易次数", f"{result.total_trades}")

    st.markdown("---")

    # 权益曲线图
    st.markdown("### 📈 权益曲线")

    if result.equity_curve is not None:
        fig = create_equity_chart(result)
        st.plotly_chart(fig, use_container_width=True)

    # 回撤曲线
    st.markdown("### 📉 回撤曲线")
    if result.equity_curve is not None:
        fig_dd = create_drawdown_chart(result)
        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # 详细统计
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 收益统计")
        stats_df = pd.DataFrame({
            "指标": [
                "累计收益率", "年化收益率", "基准收益率", "超额收益",
                "波动率", "下行波动率", "最大回撤", "最大回撤持续天数"
            ],
            "数值": [
                f"{result.total_return*100:.2f}%",
                f"{result.annual_return*100:.2f}%",
                f"{result.benchmark_return*100:.2f}%",
                f"{result.excess_return*100:.2f}%",
                f"{result.volatility*100:.2f}%",
                f"{result.downside_volatility*100:.2f}%",
                f"{result.max_drawdown*100:.2f}%",
                f"{result.max_drawdown_duration}天"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 📊 交易统计")
        trade_df = pd.DataFrame({
            "指标": [
                "总交易次数", "盈利次数", "亏损次数", "胜率",
                "盈亏比", "平均盈利", "平均亏损", "平均持仓天数"
            ],
            "数值": [
                f"{result.total_trades}",
                f"{result.win_trades}",
                f"{result.lose_trades}",
                f"{result.win_rate*100:.1f}%",
                f"{result.profit_loss_ratio:.2f}",
                f"¥{result.avg_win:,.0f}",
                f"¥{result.avg_loss:,.0f}",
                f"{result.avg_holding_days:.1f}天"
            ]
        })
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 交易记录
    st.markdown("### 📋 交易记录")

    if result.trades:
        trades_data = []
        for t in result.trades:
            trades_data.append({
                "日期": t.date,
                "代码": t.code,
                "方向": "买入" if t.direction == "BUY" else "卖出",
                "价格": f"{t.price:.3f}",
                "股数": t.shares,
                "金额": f"¥{t.amount:,.0f}",
                "手续费": f"¥{t.commission:.2f}",
                "盈亏": f"¥{t.pnl:,.0f}" if t.direction == "SELL" else "-",
                "盈亏%": f"{t.pnl_pct*100:+.2f}%" if t.direction == "SELL" else "-",
                "备注": t.reason
            })

        trades_df = pd.DataFrame(trades_data)

        # 筛选
        col1, col2 = st.columns(2)
        with col1:
            direction_filter = st.selectbox("方向筛选", ["全部", "买入", "卖出"])
        with col2:
            code_filter = st.selectbox("标的筛选", ["全部"] + list(set([t.code for t in result.trades])))

        if direction_filter != "全部":
            trades_df = trades_df[trades_df["方向"] == direction_filter]
        if code_filter != "全部":
            trades_df = trades_df[trades_df["代码"] == code_filter]

        st.dataframe(trades_df, use_container_width=True, hide_index=True)

        # 下载按钮
        csv = trades_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 下载交易记录",
            csv,
            "trades.csv",
            "text/csv"
        )


def create_equity_chart(result):
    """创建权益曲线图"""
    df = result.equity_curve.reset_index()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("累计收益率", "每日收益率")
    )

    # 累计收益率
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_return"] * 100,
            mode="lines",
            name="策略收益",
            line=dict(color="#1f77b4", width=2)
        ),
        row=1, col=1
    )

    # 每日收益率柱状图
    colors = ["#00c853" if r >= 0 else "#ff1744" for r in df["return"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["return"] * 100,
            name="每日收益",
            marker_color=colors
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    fig.update_yaxes(title_text="收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="日收益 (%)", row=2, col=1)

    return fig


def create_drawdown_chart(result):
    """创建回撤曲线图"""
    df = result.equity_curve.reset_index()

    # 计算回撤
    rolling_max = df["total_value"].cummax()
    drawdown = (df["total_value"] - rolling_max) / rolling_max * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            name="回撤",
            line=dict(color="#ff1744", width=1),
            fillcolor="rgba(255, 23, 68, 0.3)"
        )
    )

    fig.update_layout(
        height=250,
        showlegend=False,
        hovermode="x unified",
        yaxis_title="回撤 (%)"
    )

    return fig
