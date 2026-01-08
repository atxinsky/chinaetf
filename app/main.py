# coding=utf-8
"""
ETF量化回测系统 - Streamlit主入口
"""

import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import STREAMLIT_PORT

# 页面配置
st.set_page_config(
    page_title="ETF量化回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    # 侧边栏导航
    st.sidebar.title("📊 ETF量化系统")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "导航",
        ["🏠 首页", "📈 回测系统", "📊 数据管理", "⚙️ 系统设置"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("版本: 1.0.0")

    # 页面路由
    if page == "🏠 首页":
        show_home()
    elif page == "📈 回测系统":
        show_backtest()
    elif page == "📊 数据管理":
        show_data_manage()
    elif page == "⚙️ 系统设置":
        show_settings()


def show_home():
    """首页"""
    st.markdown('<p class="main-header">ETF量化回测系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于AKShare数据的ETF趋势轮动策略回测平台</p>', unsafe_allow_html=True)

    # 快捷入口
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🚀 快速回测")
        st.write("使用BigBrother V14策略进行回测")
        if st.button("开始回测", key="quick_backtest"):
            st.session_state["page"] = "backtest"
            st.rerun()

    with col2:
        st.markdown("### 📊 数据更新")
        st.write("更新ETF历史数据")
        if st.button("更新数据", key="update_data"):
            st.session_state["page"] = "data"
            st.rerun()

    with col3:
        st.markdown("### 📋 策略库")
        st.write("查看和管理策略")
        st.button("策略管理", key="strategy_manage")

    st.markdown("---")

    # 系统概览
    st.markdown("### 📈 系统概览")

    col1, col2, col3, col4 = st.columns(4)

    try:
        from core.data_service import get_data_service
        ds = get_data_service()
        info = ds.get_data_info()

        with col1:
            st.metric("ETF数量", f"{len(info)}个")
        with col2:
            if len(info) > 0:
                total_rows = info["rows"].sum()
                st.metric("数据条数", f"{total_rows:,}")
            else:
                st.metric("数据条数", "0")
        with col3:
            if len(info) > 0:
                st.metric("最早日期", info["start_date"].min())
            else:
                st.metric("最早日期", "N/A")
        with col4:
            if len(info) > 0:
                st.metric("最新日期", info["end_date"].max())
            else:
                st.metric("最新日期", "N/A")

        # 数据统计表
        if len(info) > 0:
            st.markdown("### 📋 数据统计")
            st.dataframe(info, use_container_width=True)

    except Exception as e:
        st.warning(f"无法加载数据统计: {e}")

    # 策略说明
    st.markdown("---")
    st.markdown("### 📖 BigBrother V14 策略说明")

    with st.expander("查看策略详情", expanded=False):
        st.markdown("""
        **BigBrother V14** 是一个ETF趋势轮动策略，核心逻辑：

        **入场条件：**
        - EMA(20) 金叉 EMA(60)
        - ADX > 20（趋势强度）
        - 价格接近20日高点（>95%）
        - A股大盘非熊市（海外标的豁免）

        **出场条件：**
        - 硬止损 7%
        - ATR止损（2.5倍ATR）
        - 追踪止盈（15%触发，6%回撤）
        - 均线死叉
        - 持仓超120天且收益<3%

        **标的池：**
        - 513100.SH 纳指ETF
        - 513050.SH 中概互联
        - 512480.SH 半导体ETF
        - 515030.SH 新能车ETF
        - 518880.SH 黄金ETF
        - 512890.SH 红利低波
        - 588000.SH 科创50
        - 516010.SH 游戏动漫
        """)


def show_backtest():
    """回测页面"""
    from app.pages.backtest import render_backtest_page
    render_backtest_page()


def show_data_manage():
    """数据管理页面"""
    from app.pages.data_manage import render_data_page
    render_data_page()


def show_settings():
    """系统设置页面"""
    st.markdown("## ⚙️ 系统设置")

    # 数据库设置
    st.markdown("### 💾 数据库设置")
    from config import DATABASE_PATH
    st.text_input("数据库路径", value=str(DATABASE_PATH), disabled=True)

    # 回测默认参数
    st.markdown("### 📊 回测默认参数")
    from config import DEFAULT_BACKTEST_CONFIG

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("初始资金", value=DEFAULT_BACKTEST_CONFIG["initial_capital"], disabled=True)
        st.number_input("手续费率", value=DEFAULT_BACKTEST_CONFIG["commission_rate"], format="%.4f", disabled=True)
    with col2:
        st.number_input("滑点", value=DEFAULT_BACKTEST_CONFIG["slippage"], format="%.4f", disabled=True)
        st.text_input("基准", value=DEFAULT_BACKTEST_CONFIG["benchmark"], disabled=True)

    # ETF池配置
    st.markdown("### 📋 ETF池配置")
    from config import ETF_POOLS

    for category, etfs in ETF_POOLS.items():
        with st.expander(f"{category} ({len(etfs)}个)"):
            for code, name in etfs.items():
                st.text(f"{code} - {name}")


if __name__ == "__main__":
    main()
