# ETF量化回测系统

基于AKShare数据的ETF趋势轮动策略回测平台。

## 功能特性

- **数据管理**: AKShare获取ETF历史数据，SQLite本地存储
- **技术指标**: EMA、ADX、ATR、RSI、MACD等
- **回测引擎**: 支持多标的组合回测，完整绩效指标
- **策略框架**: 可扩展的策略基类，内置BigBrother V14策略
- **Web界面**: Streamlit实现，复刻BigQuant风格
- **Docker支持**: 一键部署

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动Web界面

```bash
python run.py web
```

访问: http://localhost:8505

### 更新数据

```bash
python run.py update
```

### 命令行回测

```bash
python run.py backtest
```

## Docker部署

```bash
docker-compose up -d
```

## 项目结构

```
ETF量化/
├── app/                    # Streamlit应用
│   ├── main.py            # 主入口
│   └── pages/             # 页面
├── core/                   # 核心模块
│   ├── data_service.py    # 数据服务
│   ├── backtest_engine.py # 回测引擎
│   └── indicators.py      # 技术指标
├── strategies/             # 策略
│   ├── base.py            # 策略基类
│   └── bigbrother_v14.py  # BigBrother V14
├── data/                   # 数据目录
├── config.py              # 配置文件
├── run.py                 # 启动入口
└── requirements.txt       # 依赖
```

## BigBrother V14 策略

ETF趋势轮动策略，核心逻辑：

**入场条件:**
- EMA(20) 金叉 EMA(60)
- ADX > 20
- 价格接近20日高点

**出场条件:**
- 硬止损 7%
- ATR止损 2.5倍
- 追踪止盈 (15%触发, 6%回撤)
- 均线死叉

**默认标的池:**
- 513100.SH 纳指ETF
- 513050.SH 中概互联
- 512480.SH 半导体ETF
- 515030.SH 新能车ETF
- 518880.SH 黄金ETF
- 512890.SH 红利低波
- 588000.SH 科创50
- 516010.SH 游戏动漫

## License

MIT
