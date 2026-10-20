# 🏆 Aurum Intelligence | 长征黄金量化研究平台

> 面向 XAU/USD 现货黄金的多源数据 · 多因子量化 · AI 解读一体化研究平台

一个基于 **Streamlit** 的黄金量化分析系统。所有数值均由**本地量化引擎**（`pandas` 因子计算 + 向量化回测）产出，大模型只负责自然语言解读、**不参与任何数字生成**；没有 API Key 或接口异常时自动降级为本地规则化结论，页面永不崩。

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FFDD00?style=flat&logo=streamlit&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ 核心特性

- **📊 三条独立数据链**：黄金行情（XAUS）、宏观经济指标（FRED）、COMEX 黄金库存（东方财富日频 + metalcharts 周频），任一环节失败互不影响。
- **🧮 10 因子多因子打分引擎**：技术面 / 宏观面 / 实物面三域因子加权，权重页面侧边可调，因子缺失时按可用权重重新归一化。
- **🛡️ 四级库存核验体系**：`VERIFIED` / `ACCEPTED` / `REVIEW` / `REJECTED`，跨源交叉验证 + 日期口径自检，量化打分只采用可信记录。
- **📈 轻量向量化回测**：仓位 `shift(1)` 杜绝未来函数，交易成本按换手双边计提，同时输出策略与「买入持有」基准对照。
- **🤖 AI 解读 + 本地降级**：DeepSeek 大模型基于本地事实简报生成 Markdown 解读；无密钥 / 超时 / 报错时降级为本地规则化结论。
- **🖥️ 六页一体化看板**：市场总览、行情分析、AI 量化分析、基本面分析、COMEX 库存分析、数据中心。
- **⏰ 每日自动更新**：一条命令跑完三条链，可注册为 Windows 计划任务，幂等、步骤隔离、运行留痕。

---

## 🧭 页面导航

| 页面 | 说明 |
| --- | --- |
| **市场总览** | 最新收盘价、RSI14、MA20 与价格/均线走势 |
| **行情分析** | K线 + 均线、RSI14、MACD 三联技术分析 |
| **AI 量化分析** | 多因子综合信号、因子归因、回测绩效、AI 解读 |
| **基本面分析** | FRED 宏观指标总览、走势、黄金与实际利率相关性 |
| **COMEX 库存分析** | 库存历史走势与人工核验工作台 |
| **数据中心** | 三类原始数据明细浏览与 CSV 导出 |

---

## 🏗️ 项目结构

```
AI量化分析/
├── app.py                      # Streamlit 多页入口（6 页路由 + 缓存 + 数据加载）
├── requirements.txt            # 依赖清单
├── .env.example.py             # 环境变量模板
├── config/
│   └── settings.py             # 全局配置
├── database/
│   └── db.py                   # SQLite 建表 / 旧库幂等补列 / 连接管理
├── src/
│   ├── data_sources/           # 数据采集层
│   │   ├── yahoo_finance.py        # 黄金行情源
│   │   ├── macro_data.py           # FRED 宏观指标
│   │   ├── comex_inventory_api.py  # COMEX 库存采集 + 跨源验证 + 核验
│   │   ├── gold_api.py             # 行情接口封装
│   │   └── deepseek_web_search.py  # DeepSeek 联网搜索
│   ├── data_processing/        # 数据清洗入库层
│   │   ├── gold_pipeline.py        # 行情：获取→验证→入库
│   │   ├── macro_pipeline.py       # 宏观：获取→检查→入库
│   │   ├── technical_indicators.py # 技术指标
│   │   └── validator.py            # 数据校验
│   ├── quant/                  # 量化核心
│   │   ├── factors.py              # 10 因子打分引擎
│   │   ├── backtest.py             # 向量化回测
│   │   ├── ai_analyst.py           # AI 解读 + 本地降级
│   │   └── dashboard.py            # AI 量化页面渲染
│   └── ui/
│       └── comex_page.py           # COMEX 库存页面与核验工作台
├── scripts/
│   ├── daily_update.py             # 每日更新统一入口
│   └── register_daily_task.ps1     # Windows 计划任务注册
├── tests/                      # pytest 测试（130 项）
├── docs/                       # 数据链路与设计文档
└── data/                       # SQLite 数据库与运行留痕
```

---

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- Windows（计划任务脚本为 PowerShell；核心功能跨平台）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example.py` 内容为项目根目录下的 `.env` 文件，按需填写：

```ini
# 大模型 API 配置（AI 解读，可留空 → 自动降级为本地规则结论）
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat

# 宏观经济数据源（FRED 官方 API）
FRED_API_KEY=

# 行情数据源
GOLD_API_KEY=

# 项目运行环境
APP_ENV=development
```

> ⚠️ `.env` 已被 `.gitignore` 忽略，请勿提交密钥。

### 4. 初始化数据库

```bash
python database/db.py
```

自动创建 `data/aurum.db` 及以下数据表（幂等，旧库只增量补列不丢数据）：

| 表 | 用途 |
| --- | --- |
| `gold_prices` | 黄金历史行情（OHLCV） |
| `macro_data` | 宏观经济指标观测值 |
| `comex_inventory` | COMEX 黄金库存 |
| `comex_inventory_quality` | 库存核验状态与审计 |
| `institution_reports` | 机构观点 |

### 5. 更新数据

```bash
# 三条数据链全部更新（行情 / 宏观 / 库存）
python -m scripts.daily_update --commit

# 只更新 COMEX 库存
python -m scripts.daily_update --commit --only comex

# 预览模式（不写库）
python -m scripts.daily_update
```

### 6. 启动 Web 应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可。

---

## 🧮 量化因子体系

10 个因子分为三域，综合得分范围 **-1（极度偏空）~ +1（极度偏多）**，全部数值本地计算：

| 域 | 因子 | 默认权重 | 含义（+1 方向） |
| --- | --- | --- | --- |
| 技术面 | `f_trend_ma` | 1.5 | MA20 在 MA50 之上，中期趋势偏多 |
| 技术面 | `f_price_ma` | 1.0 | 收盘价站上 MA20，短期结构偏多 |
| 技术面 | `f_macd` | 1.0 | MACD 柱为正且走高，动能偏多 |
| 技术面 | `f_momentum_20` | 1.0 | 20 日涨幅相对自身波动率偏高 |
| 技术面 | `f_rsi` | 0.8 | RSI14 处于超卖区，存在反弹空间 |
| 宏观面 | `f_real_rate` | 1.5 | 实际利率下行，持金机会成本降低 |
| 宏观面 | `f_nominal_rate` | 0.8 | 名义利率下行，对金价形成支撑 |
| 宏观面 | `f_inflation` | 0.8 | 通胀高于常态，抗通胀需求上升 |
| 宏观面 | `f_fedfunds` | 0.7 | 政策利率下行，流动性偏宽松 |
| 实物面 | `f_inventory` | 0.7 | COMEX 总库存环比下降，实物交割偏强 |

**关键设计约束：**
- 宏观与库存数据一律按「公布可用日」对齐（`merge_asof` backward），杜绝未来函数（look-ahead bias）。
- 各因子设中性死区，避免噪声频繁翻转信号。
- 库存因子采用 `total` 口径，7 个日历天回看窗口，语义在日频/周频源之间保持一致。

---

## 🛡️ COMEX 库存数据链路

库存数据是本项目的重点与难点，完整设计见 [`docs/COMEX库存数据链路说明.md`](docs/COMEX库存数据链路说明.md)。

**数据来源：**

| 来源 | 频率 | 字段 | 说明 |
| --- | --- | --- | --- |
| `eastmoney` | 日频 | 仅 `total` | 主源，覆盖长、密度高 |
| `metalcharts` | 周频 | `registered` / `eligible` / `total` | 用于交叉验证与补空 |

**四级核验状态（采集程序自动判定，页面可人工干预）：**

| 状态 | 含义 | 参与打分 |
| --- | --- | --- |
| `VERIFIED` | 与第二来源在同一 CME 业务日数值一致（偏差 ≤ 0.01%） | ✅ |
| `ACCEPTED` | 单源数据，通过全部自动校验且无告警 | ✅ |
| `REVIEW` | 存在告警（跨源分歧 / 极端变化 / 周末日期 / 校验未通过） | ❌ |
| `REJECTED` | 已证伪，保留审计但不参与任何分析 | ❌ |

> 日期口径、偏移自检、跨源 ±1 天窗口等「踩坑记录」均在文档中详述，改代码前建议先读完。

---

## ⏰ 注册每日自动更新（Windows 计划任务）

```powershell
# 注册：每日 08:30 自动执行
powershell -File scripts\register_daily_task.ps1

# 指定时间
powershell -File scripts\register_daily_task.ps1 -Time "18:00"

# 查看状态
powershell -File scripts\register_daily_task.ps1 -Action show

# 注销
powershell -File scripts\register_daily_task.ps1 -Action unregister
```

任务特性：**幂等**（唯一键去重，重跑/补跑安全）、**步骤隔离**（单步失败只记录并继续）、**失败重试 3 次**（间隔 10 分钟）、**运行留痕**（结果写入 `data/daily_update/last_run.json`）。

> 注册计划任务会修改系统状态，需自行执行，本项目不会自动注册。

---

## 🧪 测试

```bash
pytest
```

覆盖数据采集、核验逻辑、量化因子与页面逻辑共 **130 项测试**：

- `test_comex_inventory.py` — 采集模块（44 项）
- `test_comex_page.py` — 页面逻辑（26 项）
- `test_quant.py` — 量化引擎（56 项，含库存因子与多来源回归）
- `test_database.py` / `test_validator.py` / `test_network.py`

---

## 📦 技术栈

| 类别 | 技术 |
| --- | --- |
| Web 框架 | Streamlit |
| 数据处理 | pandas、numpy |
| 可视化 | Plotly |
| 网络采集 | requests、beautifulsoup4 |
| 配置管理 | python-dotenv |
| 数据库 | SQLite |
| AI 解读 | DeepSeek API |
| 测试 | pytest |

---

## ⚠️ 免责声明

本平台用于**金融数据研究和历史分析**。相关性不代表因果关系，历史表现不代表未来结果。数据可能存在发布时间差异、延迟或修订。AI 生成的解读内容仅供参考，**不构成任何投资建议**；市场有风险，决策需独立判断。
