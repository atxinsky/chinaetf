# coding=utf-8
"""
ETF数据服务层
支持从AKShare获取数据，并存储到SQLite数据库
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ETFDataService:
    """
    ETF数据服务

    功能：
    1. 从AKShare获取ETF日线数据
    2. SQLite本地存储
    3. 数据缓存和增量更新
    4. 技术指标计算
    """

    def __init__(self, db_path: str = None):
        """
        初始化数据服务

        Args:
            db_path: 数据库路径，默认为config中的路径
        """
        if db_path is None:
            from config import DATABASE_PATH
            db_path = str(DATABASE_PATH)

        self.db_path = db_path
        self._init_database()
        self._cache = {}  # 内存缓存

    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ETF日线数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                UNIQUE(code, date)
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_etf_daily_code_date
            ON etf_daily(code, date)
        """)

        # ETF基本信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_info (
                code TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                last_update TEXT
            )
        """)

        # 数据更新记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                update_time TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                rows_added INTEGER
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")

    def fetch_from_akshare(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从AKShare获取ETF日线数据

        Args:
            code: ETF代码 (如 '513100.SH')
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        try:
            import akshare as ak
        except ImportError:
            logger.error("请安装akshare: pip install akshare")
            raise ImportError("akshare未安装，请运行: pip install akshare")

        # 转换代码格式
        ak_code = code.split(".")[0]

        try:
            # 使用AKShare获取ETF数据
            # fund_etf_hist_em 是东方财富的ETF历史数据接口
            df = ak.fund_etf_hist_em(
                symbol=ak_code,
                period="daily",
                start_date=start_date.replace("-", "") if start_date else "20190101",
                end_date=end_date.replace("-", "") if end_date else datetime.now().strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )

            if df is None or len(df) == 0:
                logger.warning(f"AKShare返回空数据: {code}")
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            })

            # 确保日期格式
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            # 选择需要的列
            columns = ["date", "open", "high", "low", "close", "volume", "amount"]
            df = df[[col for col in columns if col in df.columns]]

            logger.info(f"从AKShare获取数据成功: {code}, {len(df)}条记录")
            return df

        except Exception as e:
            logger.error(f"从AKShare获取数据失败: {code}, 错误: {e}")
            return pd.DataFrame()

    def save_to_db(self, code: str, df: pd.DataFrame) -> int:
        """
        保存数据到数据库

        Args:
            code: ETF代码
            df: 数据DataFrame

        Returns:
            新增记录数
        """
        if df is None or len(df) == 0:
            return 0

        conn = sqlite3.connect(self.db_path)

        # 添加code列
        df = df.copy()
        df["code"] = code

        # 使用INSERT OR IGNORE避免重复
        rows_before = pd.read_sql("SELECT COUNT(*) as cnt FROM etf_daily WHERE code=?",
                                   conn, params=[code]).iloc[0]["cnt"]

        df.to_sql("etf_daily", conn, if_exists="append", index=False,
                  method=lambda table, conn, keys, data_iter:
                  conn.executemany(
                      f"INSERT OR IGNORE INTO {table.name} ({','.join(keys)}) VALUES ({','.join(['?']*len(keys))})",
                      list(data_iter)
                  ))

        rows_after = pd.read_sql("SELECT COUNT(*) as cnt FROM etf_daily WHERE code=?",
                                  conn, params=[code]).iloc[0]["cnt"]

        rows_added = rows_after - rows_before

        # 记录更新日志
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO update_log (code, update_time, start_date, end_date, rows_added)
            VALUES (?, ?, ?, ?, ?)
        """, (code, datetime.now().isoformat(),
              df["date"].min(), df["date"].max(), rows_added))

        conn.commit()
        conn.close()

        # 清除缓存
        if code in self._cache:
            del self._cache[code]

        logger.info(f"保存数据到数据库: {code}, 新增{rows_added}条")
        return rows_added

    def get_data(self, code: str, start_date: str = None, end_date: str = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        获取ETF数据（优先从本地数据库，没有则从网络获取）

        Args:
            code: ETF代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            DataFrame
        """
        cache_key = f"{code}_{start_date}_{end_date}"

        # 检查缓存
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        conn = sqlite3.connect(self.db_path)

        # 构建查询
        query = "SELECT date, open, high, low, close, volume, amount FROM etf_daily WHERE code=?"
        params = [code]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # 如果本地没有数据，从网络获取
        if len(df) == 0:
            logger.info(f"本地无数据，从网络获取: {code}")
            df = self.fetch_from_akshare(code, start_date, end_date)
            if len(df) > 0:
                self.save_to_db(code, df)

        # 缓存
        if use_cache and len(df) > 0:
            self._cache[cache_key] = df.copy()

        return df

    def update_data(self, code: str, force: bool = False) -> int:
        """
        更新ETF数据（增量更新）

        Args:
            code: ETF代码
            force: 是否强制全量更新

        Returns:
            新增记录数
        """
        conn = sqlite3.connect(self.db_path)

        if force:
            # 强制全量更新
            cursor = conn.cursor()
            cursor.execute("DELETE FROM etf_daily WHERE code=?", [code])
            conn.commit()
            start_date = "2019-01-01"
        else:
            # 获取最后更新日期
            result = pd.read_sql(
                "SELECT MAX(date) as last_date FROM etf_daily WHERE code=?",
                conn, params=[code]
            )
            last_date = result.iloc[0]["last_date"]

            if last_date:
                # 从最后日期的下一天开始
                start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = "2019-01-01"

        conn.close()

        # 获取新数据
        end_date = datetime.now().strftime("%Y-%m-%d")
        df = self.fetch_from_akshare(code, start_date, end_date)

        if len(df) > 0:
            return self.save_to_db(code, df)
        return 0

    def update_all(self, codes: List[str] = None, force: bool = False) -> Dict[str, int]:
        """
        批量更新ETF数据

        Args:
            codes: ETF代码列表，默认更新所有配置的ETF
            force: 是否强制全量更新

        Returns:
            {code: rows_added}
        """
        if codes is None:
            from config import ALL_ETFS
            codes = list(ALL_ETFS.keys())

        results = {}
        for i, code in enumerate(codes):
            logger.info(f"更新数据 [{i+1}/{len(codes)}]: {code}")
            try:
                rows = self.update_data(code, force)
                results[code] = rows
                # 避免请求过快
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"更新失败: {code}, 错误: {e}")
                results[code] = -1

        return results

    def get_data_with_indicators(self, code: str, start_date: str = None,
                                  end_date: str = None) -> pd.DataFrame:
        """
        获取带技术指标的数据

        Args:
            code: ETF代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            带技术指标的DataFrame
        """
        from .indicators import calculate_indicators

        # 获取更早的数据用于计算指标
        actual_start = start_date
        if start_date:
            # 多获取90天数据用于指标预热
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            extended_start = (dt - timedelta(days=120)).strftime("%Y-%m-%d")
        else:
            extended_start = None

        df = self.get_data(code, extended_start, end_date)

        if len(df) == 0:
            return df

        # 计算技术指标
        df = calculate_indicators(df)

        # 过滤到实际需要的日期范围
        if actual_start:
            df = df[df["date"] >= actual_start]

        return df.reset_index(drop=True)

    def get_multiple_data(self, codes: List[str], start_date: str = None,
                          end_date: str = None, with_indicators: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取多个ETF的数据

        Args:
            codes: ETF代码列表
            start_date: 开始日期
            end_date: 结束日期
            with_indicators: 是否计算技术指标

        Returns:
            {code: DataFrame}
        """
        results = {}
        for code in codes:
            if with_indicators:
                df = self.get_data_with_indicators(code, start_date, end_date)
            else:
                df = self.get_data(code, start_date, end_date)
            results[code] = df
        return results

    def get_available_codes(self) -> List[str]:
        """获取数据库中有数据的ETF代码列表"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql("SELECT DISTINCT code FROM etf_daily", conn)
        conn.close()
        return result["code"].tolist()

    def get_data_info(self) -> pd.DataFrame:
        """获取数据库中所有ETF的数据统计信息"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql("""
            SELECT
                code,
                COUNT(*) as rows,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM etf_daily
            GROUP BY code
            ORDER BY code
        """, conn)
        conn.close()
        return result

    def clear_cache(self):
        """清除内存缓存"""
        self._cache.clear()
        logger.info("缓存已清除")


# 单例模式
_data_service_instance = None

def get_data_service(db_path: str = None) -> ETFDataService:
    """获取数据服务单例"""
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = ETFDataService(db_path)
    return _data_service_instance
