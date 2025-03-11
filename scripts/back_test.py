# back_test.py
# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger(log_filename="back_test.log", log_level=logging.INFO):
    """
    设置日志记录器，输出到控制台和日志文件。
    参数:
        log_filename (str): 日志文件名
        log_level (int): 日志级别，默认 INFO
    """
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("back_test")
    logger.setLevel(log_level)
    logger.handlers.clear()
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()
logger.info("日志系统初始化成功")

# ================= 数据加载函数 =================
def load_price_data(data_dir):
    """
    从指定目录加载价格数据
    """
    price_dfs = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_cleaned_featured_standardized.csv"):
            stock_code = filename.split("_")[0]
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Stock"] = stock_code
            price_dfs.append(df)
    if not price_dfs:
        raise ValueError(f"未在 {data_dir} 中找到价格数据文件")
    return pd.concat(price_dfs, ignore_index=True)

def load_signal_data(signal_dir, timestamp=None):
    """
    从指定目录加载所有信号文件
    参数:
        signal_dir (str): 信号文件目录
        timestamp (str): 可选，指定时间戳以加载特定批次的信号文件
    """
    signal_dfs = []
    for filename in os.listdir(signal_dir):
        if filename.startswith("signal_") and filename.endswith(".csv"):
            if timestamp and timestamp not in filename:
                continue
            file_path = os.path.join(signal_dir, filename)
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            signal_dfs.append(df)
    if not signal_dfs:
        raise ValueError(f"未在 {signal_dir} 中找到信号文件")
    return pd.concat(signal_dfs, ignore_index=True)

# ================= 回测核心类 =================
class Backtest:
    def __init__(self, signal_df, price_df, initial_capital=1000000, transaction_cost=0.001, signal_threshold=0.7, max_position_size=0.5):
        """
        初始化回测对象
        参数:
            signal_df (pd.DataFrame): 包含信号的DataFrame，列包括 Date, Stock, Signal
            price_df (pd.DataFrame): 包含价格的DataFrame，列包括 Date, Stock, Open, Close 等
            initial_capital (float): 初始资金，默认为100万
            transaction_cost (float): 每笔交易成本比例，默认为0.1%
            signal_threshold (float): 触发交易的信号阈值，默认为0.7
            max_position_size (float): 每只股票最大仓位比例，默认为0.5（50%）
        """
        self.signal_df = signal_df.copy()
        self.price_df = price_df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.signal_threshold = signal_threshold
        self.max_position_size = max_position_size
        
        # 初始化持仓和资金
        self.cash = initial_capital
        self.positions = {}  # {stock: shares}
        self.portfolio_value = initial_capital
        self.trade_log = []

    def run(self):
        """运行回测"""
        logger.info("开始回测...")
        
        # 合并信号和价格数据
        self.combined_df = pd.merge(
            self.signal_df[["Date", "Stock", "Signal"]],
            self.price_df[["Date", "Stock", "Open", "Close", "Returns"]],
            on=["Date", "Stock"],
            how="inner"
        )
        if self.combined_df.empty:
            raise ValueError("信号数据和价格数据无法匹配")
        
        self.combined_df = self.combined_df.sort_values(["Date", "Stock"])
        
        # 按日期处理
        for date, daily_data in self.combined_df.groupby("Date"):
            self._process_day(date, daily_data)
        
        # 计算回测结果
        metrics = self._calculate_metrics()
        return metrics

    def _process_day(self, date, daily_data):
        """处理每一天的交易"""
        # 计算当前持仓价值
        position_value = 0
        for stock, shares in list(self.positions.items()):
            stock_data = daily_data[daily_data["Stock"] == stock]
            if not stock_data.empty:
                close_price = stock_data.iloc[0]["Close"]
                position_value += shares * close_price
            else:
                position_value += shares * self.positions.get(stock + "_last_price", 0)
        
        self.portfolio_value = self.cash + position_value
        
        for _, row in daily_data.iterrows():
            stock = row["Stock"]
            signal = row["Signal"]
            open_price = row["Open"]
            close_price = row["Close"]
            
            # 当前持仓
            current_shares = self.positions.get(stock, 0)
            
            # 计算目标持仓
            if abs(signal) >= self.signal_threshold:
                direction = 1 if signal > 0 else -1
                target_value = direction * self.portfolio_value * self.max_position_size
                target_shares = int(target_value / open_price)
            else:
                target_shares = 0
            
            # 计算交易量
            shares_to_trade = target_shares - current_shares
            
            if shares_to_trade != 0:
                trade_value = abs(shares_to_trade) * open_price
                cost = trade_value * self.transaction_cost
                
                # 检查现金是否足够
                if shares_to_trade > 0 and self.cash < trade_value + cost:
                    affordable_shares = int((self.cash - cost) / open_price)
                    shares_to_trade = affordable_shares
                    trade_value = shares_to_trade * open_price
                    cost = trade_value * self.transaction_cost
                
                # 更新现金和持仓
                self.cash -= shares_to_trade * open_price + cost
                self.positions[stock] = current_shares + shares_to_trade
                self.positions[stock + "_last_price"] = close_price
                
                # 移除空仓
                if self.positions[stock] == 0:
                    del self.positions[stock]
                    del self.positions[stock + "_last_price"]
                
                # 记录交易
                action = "BUY" if shares_to_trade > 0 else "SELL"
                self.trade_log.append({
                    "Date": date,
                    "Stock": stock,
                    "Action": action,
                    "Shares": abs(shares_to_trade),
                    "Price": open_price,
                    "Value": trade_value,
                    "Cost": cost
                })
        
        # 更新每日净值
        self.combined_df.loc[self.combined_df["Date"] == date, "Portfolio_Value"] = self.portfolio_value

    def _calculate_metrics(self):
        """计算回测绩效指标"""
        portfolio_values = self.combined_df.groupby("Date")["Portfolio_Value"].mean()
        returns = portfolio_values.pct_change().dropna()
        
        # 计算累计收益
        cumulative_return = (self.portfolio_value / self.initial_capital) - 1
        
        # 计算年化收益
        days = (self.combined_df["Date"].max() - self.combined_df["Date"].min()).days
        annualized_return = (1 + cumulative_return) ** (365.0 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        rolling_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # 交易统计
        total_trades = len(self.trade_log)
        total_cost = sum(trade["Cost"] for trade in self.trade_log)
        
        metrics = {
            "Cumulative_Return": cumulative_return,
            "Annualized_Return": annualized_return,
            "Max_Drawdown": max_drawdown,
            "Sharpe_Ratio": sharpe_ratio,
            "Total_Trades": total_trades,
            "Total_Transaction_Cost": total_cost,
            "Final_Portfolio_Value": self.portfolio_value
        }
        
        return metrics

    def save_results(self, output_dir):
        """保存回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存净值曲线
        portfolio_df = self.combined_df.groupby("Date")["Portfolio_Value"].mean().reset_index()
        portfolio_path = os.path.join(output_dir, f"portfolio_value_{timestamp}.csv")
        portfolio_df.to_csv(portfolio_path, index=False)
        logger.info(f"净值曲线已保存至: {portfolio_path}")
        
        # 保存交易记录
        trade_df = pd.DataFrame(self.trade_log)
        if not trade_df.empty:
            trade_path = os.path.join(output_dir, f"trade_log_{timestamp}.csv")
            trade_df.to_csv(trade_path, index=False)
            logger.info(f"交易记录已保存至: {trade_path}")
        
        return portfolio_path, trade_path if not trade_df.empty else None

# ================= 主函数 =================
def main(signal_dir=None, price_dir=None, timestamp=None):
    """主函数，执行回测"""
    try:
        logger.info("========== 开始回测 ==========")
        
        project_root = get_project_root()
        
        # 默认信号目录
        if signal_dir is None:
            signal_dir = os.path.join(project_root, "signals")
        
        # 默认价格数据目录
        if price_dir is None:
            price_dir = os.path.join(project_root, "data", "standardized")
        
        # 默认时间戳（可选）
        if timestamp is None:
            timestamp = "20250311_175250"  # 你提供的示例时间戳
        
        # 加载信号数据
        signal_df = load_signal_data(signal_dir, timestamp)
        logger.info(f"信号数据加载完成，共 {len(signal_df)} 条记录")
        
        # 加载价格数据
        price_df = load_price_data(price_dir)
        logger.info(f"价格数据加载完成，共 {len(price_df)} 条记录")
        
        # 初始化回测
        backtest = Backtest(
            signal_df=signal_df,
            price_df=price_df,
            initial_capital=1000000,  # 初始资金100万
            transaction_cost=0.001,   # 交易成本0.1%
            signal_threshold=0.7,     # 信号阈值0.7
            max_position_size=0.5     # 每只股票最大仓位50%
        )
        
        # 运行回测
        metrics = backtest.run()
        
        # 输出结果
        logger.info("回测结果：")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        # 保存结果
        output_dir = os.path.join(project_root, "backtest_results")
        portfolio_path, trade_path = backtest.save_results(output_dir)
        
        logger.info("========== 回测完成 ==========")
        
    except Exception as e:
        logger.error(f"回测失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="回测交易策略")
    parser.add_argument("--signal_dir", type=str, default=None, help="信号文件目录")
    parser.add_argument("--price_dir", type=str, default=None, help="价格数据目录")
    parser.add_argument("--timestamp", type=str, default=None, help="信号文件时间戳")
    args = parser.parse_args()
    main(args.signal_dir, args.price_dir, args.timestamp)