import os
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import random

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "plot_strategy_interactive.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# ================= 单股票策略绘制函数（交互式） =================
def plot_single_stock_strategy(featured_csv, strategy_csv):
    """
    利用 Plotly 绘制单个股票的交互式周K线图，并叠加交易策略信号
    :param featured_csv: 特色数据文件路径，如 data/featured/AAPL_1wk_cleaned_featured.csv
    :param strategy_csv: 策略数据文件路径，如 data/strategy/AAPL_1wk_cleaned_featured_standardized_strategy.csv
    """
    try:
        df_featured = pd.read_csv(featured_csv)
        logger.info(f"成功加载特色数据文件: {featured_csv}")
    except Exception as e:
        logger.error(f"加载特色数据文件 {featured_csv} 失败: {e}")
        return

    try:
        df_strategy = pd.read_csv(strategy_csv)
        logger.info(f"成功加载策略文件: {strategy_csv}")
    except Exception as e:
        logger.error(f"加载策略文件 {strategy_csv} 失败: {e}")
        return

    # 转换日期格式
    try:
        df_featured['Date'] = pd.to_datetime(df_featured['Date'])
        df_strategy['Date'] = pd.to_datetime(df_strategy['Date'])
    except Exception as e:
        logger.error(f"转换日期格式时出错: {e}")
        return

    # 设置索引方便匹配
    df_featured.set_index('Date', inplace=True)
    df_strategy.set_index('Date', inplace=True)

    # 提取买入与卖出信号
    if 'Strategy' not in df_strategy.columns:
        logger.error(f"策略文件 {strategy_csv} 中缺少 'Strategy' 列")
        return

    buy_signals = df_strategy[df_strategy['Strategy'] == 'Buy'].index
    sell_signals = df_strategy[df_strategy['Strategy'] == 'Sell'].index

    try:
        # 为直观显示，买入信号在最低价附近，卖出信号在最高价附近
        buy_prices = df_featured.loc[buy_signals, 'Low'] * 0.995
        sell_prices = df_featured.loc[sell_signals, 'High'] * 1.005
    except Exception as e:
        logger.error(f"提取买入/卖出价格时出错: {e}")
        return

    # 构造 candlestick 图（K线图）
    candle = go.Candlestick(x=df_featured.index,
                            open=df_featured['Open'],
                            high=df_featured['High'],
                            low=df_featured['Low'],
                            close=df_featured['Close'],
                            name="K线")
    # 构造买入信号散点图
    scatter_buy = go.Scatter(x=buy_signals,
                             y=buy_prices,
                             mode='markers',
                             marker=dict(symbol='triangle-up',
                                         color='green',
                                         size=12),
                             name="买入信号")
    # 构造卖出信号散点图
    scatter_sell = go.Scatter(x=sell_signals,
                              y=sell_prices,
                              mode='markers',
                              marker=dict(symbol='triangle-down',
                                          color='red',
                                          size=12),
                              name="卖出信号")

    data = [candle, scatter_buy, scatter_sell]

    # 设置布局：使用支持中文的字体（例如 Microsoft YaHei），同时启用交互缩放
    layout = go.Layout(
        title={
            'text': f"K线图及交易策略信号 - {os.path.basename(featured_csv)}",
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title="日期", rangeslider=dict(visible=False)),
        yaxis=dict(title="价格"),
        font=dict(family="Microsoft YaHei", size=12),
        hovermode="x unified"
    )

    fig = go.Figure(data=data, layout=layout)
    try:
        fig.show()  # 交互式图表，可缩放、平移
        logger.info(f"成功绘制股票 {os.path.basename(featured_csv)} 的交互式K线图及交易信号")
    except Exception as e:
        logger.error(f"绘制交互式图表时出错: {e}")

# ================= 批量绘制策略图（交互式） =================
def plot_all_strategies():
    """
    扫描策略目录下所有 CSV 文件，为每个股票绘制对应的交互式K线图及交易策略信号
    策略文件命名格式假设为：股票代码_1wk_cleaned_featured_standardized_strategy.csv
    对应的特色数据文件为：股票代码_1wk_cleaned_featured.csv
    """
    project_root = get_project_root()
    strategy_dir = os.path.join(project_root, "data", "strategy")
    featured_dir = os.path.join(project_root, "data", "featured")
    
    if not os.path.exists(strategy_dir):
        logger.error(f"策略目录不存在: {strategy_dir}")
        return
    if not os.path.exists(featured_dir):
        logger.error(f"特色数据目录不存在: {featured_dir}")
        return

    strategy_files = [f for f in os.listdir(strategy_dir) if f.endswith("_standardized_strategy.csv")]
    if not strategy_files:
        logger.error(f"策略目录 {strategy_dir} 中未找到策略 CSV 文件")
        return

    # 随机挑选 5 个策略文件
    selected_files = random.sample(strategy_files, min(5, len(strategy_files)))

    for strat_file in selected_files:
        # 假设文件名格式：{stockcode}_1wk_cleaned_featured_standardized_strategy.csv
        stock_code = strat_file.split("_")[0]
        featured_file = os.path.join(featured_dir, f"{stock_code}_1wk_cleaned_featured.csv")
        strat_file_path = os.path.join(strategy_dir, strat_file)
        
        if not os.path.exists(featured_file):
            logger.error(f"对应的特色数据文件不存在: {featured_file} (股票代码: {stock_code})")
            continue
        
        logger.info(f"绘制股票 {stock_code} 的策略图：特色数据文件: {featured_file}，策略文件: {strat_file_path}")
        plot_single_stock_strategy(featured_file, strat_file_path)

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始绘制交互式交易策略图 ==========")
        plot_all_strategies()
        logger.info("========== 绘制完成 ==========")
    except Exception as e:
        logger.error(f"绘图过程中发生错误: {e}")

if __name__ == "__main__":
    main()