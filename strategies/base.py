# coding=utf-8
"""
策略基类
定义策略接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    策略基类

    所有策略都需要继承此类并实现以下方法:
    - initialize(): 初始化策略参数
    - handle_data(): 每日处理函数
    """

    # 策略名称
    name: str = "BaseStrategy"

    # 策略描述
    description: str = ""

    # 策略参数定义（用于UI展示）
    parameters: Dict[str, dict] = {}

    def __init__(self, **kwargs):
        """
        初始化策略

        Args:
            **kwargs: 策略参数
        """
        self.params = kwargs
        self._context = None

    def set_context(self, context):
        """设置回测上下文"""
        self._context = context

    @property
    def context(self):
        return self._context

    @abstractmethod
    def initialize(self, context):
        """
        策略初始化（在回测开始时调用一次）

        Args:
            context: 回测上下文
        """
        pass

    @abstractmethod
    def handle_data(self, context, data):
        """
        每日数据处理（每个交易日调用一次）

        Args:
            context: 回测上下文
            data: 当日数据
        """
        pass

    def before_trading(self, context, data):
        """
        盘前处理（可选实现）

        Args:
            context: 回测上下文
            data: 当日数据
        """
        pass

    def after_trading(self, context, data):
        """
        盘后处理（可选实现）

        Args:
            context: 回测上下文
            data: 当日数据
        """
        pass

    def on_order(self, context, order):
        """
        订单回调（可选实现）

        Args:
            context: 回测上下文
            order: 订单信息
        """
        pass

    def on_trade(self, context, trade):
        """
        成交回调（可选实现）

        Args:
            context: 回测上下文
            trade: 成交信息
        """
        pass

    def get_parameter_config(self) -> Dict[str, dict]:
        """
        获取参数配置（用于UI展示）

        Returns:
            参数配置字典
        """
        return self.parameters

    def get_parameter_values(self) -> Dict[str, Any]:
        """
        获取当前参数值

        Returns:
            参数值字典
        """
        return self.params.copy()

    def set_parameter(self, name: str, value: Any):
        """
        设置参数值

        Args:
            name: 参数名
            value: 参数值
        """
        self.params[name] = value

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        获取默认参数值

        Returns:
            默认参数字典
        """
        defaults = {}
        for name, config in cls.parameters.items():
            defaults[name] = config.get("default")
        return defaults


def create_strategy(strategy_class, **kwargs) -> BaseStrategy:
    """
    工厂函数：创建策略实例

    Args:
        strategy_class: 策略类
        **kwargs: 策略参数

    Returns:
        策略实例
    """
    # 合并默认参数
    params = strategy_class.get_default_params()
    params.update(kwargs)

    return strategy_class(**params)


# 策略注册表
_strategy_registry: Dict[str, type] = {}


def register_strategy(cls):
    """
    策略注册装饰器

    Usage:
        @register_strategy
        class MyStrategy(BaseStrategy):
            ...
    """
    _strategy_registry[cls.name] = cls
    return cls


def get_strategy_class(name: str) -> Optional[type]:
    """
    根据名称获取策略类

    Args:
        name: 策略名称

    Returns:
        策略类
    """
    return _strategy_registry.get(name)


def list_strategies() -> List[str]:
    """
    列出所有注册的策略

    Returns:
        策略名称列表
    """
    return list(_strategy_registry.keys())
