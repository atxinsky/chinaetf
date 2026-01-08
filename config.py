# coding=utf-8
"""
ETF量化系统配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 数据库配置
DATABASE_PATH = DATA_DIR / "etf_data.db"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Streamlit配置
STREAMLIT_PORT = 8505

# 默认回测参数
DEFAULT_BACKTEST_CONFIG = {
    "initial_capital": 1000000,
    "commission_rate": 0.0001,
    "slippage": 0.0001,
    "benchmark": "510300.SH",  # 沪深300ETF作为基准
}

# ETF池配置
ETF_POOLS = {
    "海外指数": {
        "513100.SH": "纳指ETF",
        "513500.SH": "标普500ETF",
        "159920.SZ": "恒生ETF",
        "513030.SH": "德国DAX",
    },
    "A股宽基": {
        "510300.SH": "沪深300ETF",
        "510050.SH": "上证50ETF",
        "510500.SH": "中证500ETF",
        "159915.SZ": "创业板ETF",
        "588000.SH": "科创50ETF",
    },
    "行业主题": {
        "512480.SH": "半导体ETF",
        "515030.SH": "新能车ETF",
        "512010.SH": "医药ETF",
        "159928.SZ": "消费ETF",
        "512880.SH": "证券ETF",
        "512660.SH": "军工ETF",
        "516010.SH": "游戏ETF",
    },
    "商品": {
        "518880.SH": "黄金ETF",
        "161226.SZ": "白银基金",
        "159985.SZ": "豆粕ETF",
    },
    "债券": {
        "511260.SH": "十年国债",
        "511010.SH": "国债ETF",
    },
    "防守型": {
        "512890.SH": "红利低波",
        "513050.SH": "中概互联",
    }
}

# 所有ETF代码
ALL_ETFS = {}
for category, etfs in ETF_POOLS.items():
    ALL_ETFS.update(etfs)

# AKShare代码转换（AKShare使用不同的代码格式）
def convert_to_akshare_code(code: str) -> str:
    """将标准代码转换为AKShare格式"""
    # 513100.SH -> 513100
    return code.split(".")[0]

def convert_from_akshare_code(code: str, market: str = None) -> str:
    """将AKShare代码转换为标准格式"""
    if market:
        return f"{code}.{market}"
    # 根据代码判断市场
    if code.startswith(("51", "56", "58", "50")):
        return f"{code}.SH"
    elif code.startswith(("15", "16")):
        return f"{code}.SZ"
    return code

# 技术指标默认参数
INDICATOR_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 60,
    "adx_period": 14,
    "atr_period": 14,
    "high_period": 20,
}
