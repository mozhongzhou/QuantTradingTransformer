# 项目架构

```
QuantTradingTransformer/
├── data/                   # 数据相关的文件
│   ├── raw/               # 原始数据（如从 yfinance 下载的股价数据）
│   │   ├── AAPL.csv
│   │   ├── GOOGL.csv
│   │   └── ...
│   ├── processed/         # 预处理后的数据（如清洗、特征工程）
│   │   ├── AAPL_processed.csv
│   │   ├── GOOGL_processed.csv
│   │   └── ...
│   └── sequences/         # 为 Transformer 模型准备的序列数据
│       ├── AAPL_sequences.npy
│       ├── GOOGL_sequences.npy
│       └── ...
├── models/                # 模型相关的文件
│   ├── transformer.py     # Transformer 模型定义
│   ├── train.py          # 模型训练脚本
│   ├── evaluate.py       # 模型评估脚本
│   └── saved_models/     # 保存训练好的模型权重
│       ├── AAPL_model.pth
│       ├── GOOGL_model.pth
│       └── ...
├── strategies/            # 交易策略相关的代码
│   ├── base_strategy.py  # 基础交易策略类
│   ├── transformer_strategy.py  # 基于 Transformer 的交易策略
│   └── backtest.py       # 回测脚本
├── notebooks/            # Jupyter Notebook 文件，用于实验和分析
│   ├── data_exploration.ipynb  # 数据探索和可视化
│   ├── model_training.ipynb    # 模型训练和调优
│   └── strategy_backtest.ipynb # 策略回测和评估
├── logs/                 # 日志文件
│   ├── training_logs/    # 模型训练日志
│   │   ├── AAPL_training.log
│   │   └── ...
│   └── backtest_logs/    # 回测日志
│       ├── AAPL_backtest.log
│       └── ...
├── results/              # 项目结果文件
│   ├── predictions/      # 模型预测结果
│   │   ├── AAPL_predictions.csv
│   │   └── ...
│   ├── backtest_reports/ # 回测报告
│   │   ├── AAPL_report.txt
│   │   └── ...
│   └── plots/            # 可视化图表
│       ├── AAPL_signals.png
│       └── ...
├── scripts/              # 主要执行脚本
│   ├── download_data.py  # 下载股价数据
│   ├── preprocess_data.py  # 预处理数据
│   ├── train_model.py    # 训练模型
│   ├── run_backtest.py   # 运行回测
│   └── visualize_results.py  # 可视化结果
├── configs/              # 配置文件
│   ├── model_config.json # 模型超参数配置
│   ├── data_config.json  # 数据处理参数配置
│   └── strategy_config.json  # 策略配置
├── requirements.txt      # 项目依赖
└── README.md
```
