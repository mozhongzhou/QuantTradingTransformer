import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import logging

# ================= 配置文件加载 =================
def load_config():
    """
    从项目根目录下的configs/back_test_config.json加载回测参数配置
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, "configs", "back_test_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise Exception(f"加载配置文件失败: {e}")

# ================= 项目路径管理 =================
def get_project_root():
    """
    获取项目根目录（即当前脚本所在目录的上一级目录）
    如果当前脚本在项目根目录下，可直接返回当前目录
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger():
    """
    配置日志记录器，将日志同时输出到控制台和日志文件
    """
    project_root = get_project_root()
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "backtest.log")
    
    logger = logging.getLogger("BacktestLogger")
    logger.setLevel(logging.DEBUG)
    
    # 创建文件处理器
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# ================= 策略回测类 =================
class StrategyBacktester:
    def __init__(self, initial_capital=1000000, buy_cash_ratio=0.9):
        """
        初始化回测器，设置初始资金和买入时的现金比例
        
        参数:
        -----------
        initial_capital: float
          回测起始资金
        buy_cash_ratio: float
          买入时使用的现金比例（例如0.9表示90%现金用于买入）
        """
        self.initial_capital = initial_capital
        self.buy_cash_ratio = buy_cash_ratio
        self.results = {}
        logger.info(f"初始化回测器，初始资金: {initial_capital}，买入比例: {buy_cash_ratio}")
    
    def load_data(self, original_data_path, strategy_path):
        """
        加载并准备回测所需的数据
        
        参数:
        -----------
        original_data_path: str
          原始价格数据CSV的相对路径
        strategy_path: str
          策略信号CSV的相对路径
        """
        try:
            self.original_data = pd.read_csv(original_data_path)
            logger.info(f"成功加载原始数据: {original_data_path}")
        except Exception as e:
            logger.error(f"加载原始数据 {original_data_path} 失败: {e}")
            raise e
        
        try:
            self.strategy_data = pd.read_csv(strategy_path)
            logger.info(f"成功加载策略数据: {strategy_path}")
        except Exception as e:
            logger.error(f"加载策略数据 {strategy_path} 失败: {e}")
            raise e
        
        try:
            self.original_data['Date'] = pd.to_datetime(self.original_data['Date'])
            self.strategy_data['Date'] = pd.to_datetime(self.strategy_data['Date'])
            logger.debug("转换Date列为datetime格式成功")
        except Exception as e:
            logger.error(f"转换日期格式失败: {e}")
            raise e
        
        try:
            # 根据Date字段合并数据，采用inner join保证数据对齐
            self.data = pd.merge(self.original_data, 
                                 self.strategy_data[['Date', 'Strategy']], 
                                 on='Date', how='inner')
            self.data = self.data.copy()
            # 对缺失策略信号填充为“Hold”
            self.data['Strategy'] = self.data['Strategy'].fillna('Hold')
            logger.info(f"合并数据后，共有 {len(self.data)} 条记录")
            logger.debug(f"策略信号分布: {self.data['Strategy'].value_counts().to_dict()}")
        except Exception as e:
            logger.error(f"数据合并失败: {e}")
            raise e

    def run_backtest(self):
        """
        对加载的数据运行回测仿真
        """
        try:
            # 初始化组合追踪变量
            self.data['Position'] = 0       # 持有股票数量
            self.data['Cash'] = self.initial_capital  # 手头现金
            self.data['Holdings'] = 0       # 持仓市值
            self.data['Portfolio_Value'] = self.initial_capital  # 组合总市值
            self.data['Returns'] = 0.0      # 每日收益率
            logger.info("初始化组合追踪变量完成")
        except Exception as e:
            logger.error(f"初始化组合变量失败: {e}")
            raise e
        
        transactions = []  # 用于记录每笔交易
        
        cash = self.initial_capital
        shares_held = 0
        
        for i in range(len(self.data)):
            try:
                date = self.data.iloc[i]['Date']
                close_price = self.data.iloc[i]['Close']
                strategy = self.data.iloc[i]['Strategy']
            except Exception as e:
                logger.error(f"读取第 {i} 条记录失败: {e}")
                continue
            
            # 根据策略执行交易
            if strategy == 'Buy' and cash > 0:
                cash_to_spend = cash * self.buy_cash_ratio  # 使用配置比例的现金买入
                shares_to_buy = cash_to_spend / close_price
                shares_held += shares_to_buy
                cash -= cash_to_spend
                
                transactions.append({
                    'Date': date,
                    'Action': 'Buy',
                    'Price': close_price,
                    'Shares': shares_to_buy,
                    'Value': cash_to_spend,
                    'Cash_Remaining': cash
                })
                logger.debug(f"{date} 买入: {shares_to_buy:.2f} 股，花费 {cash_to_spend:.2f}")
            elif strategy == 'Sell' and shares_held > 0:
                cash_from_sale = shares_held * close_price
                cash += cash_from_sale
                
                transactions.append({
                    'Date': date,
                    'Action': 'Sell',
                    'Price': close_price,
                    'Shares': shares_held,
                    'Value': cash_from_sale,
                    'Cash_Remaining': cash
                })
                logger.debug(f"{date} 卖出: {shares_held:.2f} 股，收入 {cash_from_sale:.2f}")
                shares_held = 0
            
            # 更新当日持仓市值和组合总市值
            holdings_value = shares_held * close_price
            portfolio_value = cash + holdings_value
            
            self.data.loc[self.data.index[i], 'Position'] = shares_held
            self.data.loc[self.data.index[i], 'Cash'] = cash
            self.data.loc[self.data.index[i], 'Holdings'] = holdings_value
            self.data.loc[self.data.index[i], 'Portfolio_Value'] = portfolio_value
            
            if i > 0:
                prev_value = self.data.iloc[i-1]['Portfolio_Value']
                daily_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
                self.data.loc[self.data.index[i], 'Returns'] = daily_return
        
        self.transactions = pd.DataFrame(transactions)
        logger.info(f"交易记录生成完毕，共计 {len(self.transactions)} 笔交易")
        
        # 计算买入并持有策略表现
        self.calculate_buy_and_hold()
        # 计算绩效指标
        self.calculate_performance_metrics()
        
        return self.results
    
    def calculate_buy_and_hold(self):
        """
        计算买入并持有策略表现，用于与回测策略对比
        """
        try:
            initial_price = self.data.iloc[0]['Close']
            final_price = self.data.iloc[-1]['Close']
            shares_bought = self.initial_capital / initial_price
            self.data['BuyHold_Value'] = self.data['Close'] * shares_bought
            buy_hold_return = (final_price / initial_price) - 1
            self.results['buy_hold_return'] = buy_hold_return
            logger.info(f"买入并持有策略收益: {buy_hold_return*100:.2f}%")
        except Exception as e:
            logger.error(f"计算买入并持有策略表现失败: {e}")
            raise e

    def calculate_performance_metrics(self):
        """
        计算策略的关键绩效指标，如总收益率、年化收益、最大回撤、夏普比率等
        """
        try:
            initial_value = self.initial_capital
            final_value = self.data.iloc[-1]['Portfolio_Value']
            total_return = (final_value / initial_value) - 1
        except Exception as e:
            logger.error(f"计算总收益率失败: {e}")
            total_return = 0
        
        try:
            days = (self.data.iloc[-1]['Date'] - self.data.iloc[0]['Date']).days
            years = days / 365
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        except Exception as e:
            logger.error(f"计算年化收益率失败: {e}")
            annualized_return = 0
        
        try:
            portfolio_values = self.data['Portfolio_Value'].values
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            max_drawdown = 0
        
        try:
            returns = self.data['Returns'].dropna()
            sharpe_ratio = (returns.mean() * 52) / (returns.std() * np.sqrt(52)) if returns.std() > 0 else 0
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            sharpe_ratio = 0
        
        num_trades = len(self.transactions) if hasattr(self, 'transactions') else 0
        
        # 计算胜率
        win_rate = 0
        if num_trades > 0 and hasattr(self, 'transactions'):
            self.transactions['Profit'] = np.nan
            for i, row in self.transactions[self.transactions['Action'] == 'Sell'].iterrows():
                sell_date = row['Date']
                sell_value = row['Value']
                buys_before = self.transactions[(self.transactions['Date'] < sell_date) & 
                                                (self.transactions['Action'] == 'Buy')]
                if not buys_before.empty:
                    most_recent_buy = buys_before.iloc[-1]
                    buy_value = most_recent_buy['Value']
                    profit = sell_value - buy_value
                    self.transactions.loc[i, 'Profit'] = profit
            profitable_trades = sum(self.transactions['Profit'] > 0)
            win_rate = profitable_trades / (len(self.transactions) / 2) if len(self.transactions) > 0 else 0
        
        self.results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'buy_hold_return': self.results.get('buy_hold_return', 0)
        }
        
        logger.info(f"绩效指标计算完毕，总收益率: {total_return*100:.2f}%, 年化收益: {annualized_return*100:.2f}%, 最大回撤: {max_drawdown*100:.2f}%")
        return self.results

    def plot_results(self, save_path=None):
        """
        可视化回测结果，绘制组合价值、价格走势与交易信号、持仓、收益分布等图表，
        同时在图中添加绩效指标文本
        
        参数:
        --------
        save_path: str, optional
          保存图像的相对路径（如提供，则保存图片）
        """
        try:
            plt.style.use('fivethirtyeight')
            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(3, 2, figure=fig)
            
            # 子图1：组合价值随时间变化
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(self.data['Date'], self.data['Portfolio_Value'], 'b-', linewidth=2, label='Strategy')
            ax1.plot(self.data['Date'], self.data['BuyHold_Value'], 'g--', linewidth=1, label='Buy & Hold')
            ax1.set_title('Portfolio Value Over Time', fontsize=14)
            ax1.legend()
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            ax1.grid(True)
            
            # 子图2：价格走势与交易信号
            ax2 = fig.add_subplot(gs[1, :])
            ax2.plot(self.data['Date'], self.data['Close'], 'k-', alpha=0.7, linewidth=1)
            for _, transaction in self.transactions.iterrows():
                if transaction['Action'] == 'Buy':
                    ax2.scatter(transaction['Date'], transaction['Price'], color='green', marker='^', s=100, alpha=0.7)
                elif transaction['Action'] == 'Sell':
                    ax2.scatter(transaction['Date'], transaction['Price'], color='red', marker='v', s=100, alpha=0.7)
            ax2.set_title('Trading Signals on Price', fontsize=14)
            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            ax2.grid(True)
            
            # 子图3：持仓规模随时间变化
            ax3 = fig.add_subplot(gs[2, 0])
            ax3.plot(self.data['Date'], self.data['Position'], 'b-', linewidth=1)
            ax3.set_title('Position Size', fontsize=12)
            ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            ax3.grid(True)
            
            # 子图4：每日收益率分布图
            ax4 = fig.add_subplot(gs[2, 1])
            sns.histplot(self.data['Returns'].dropna(), bins=50, kde=True, ax=ax4)
            ax4.set_title('Returns Distribution', fontsize=12)
            ax4.grid(True)
            
            # 在图中添加绩效指标文本
            performance_text = (
                f"Total Return: {self.results['total_return']*100:.2f}%\n"
                f"Annual Return: {self.results['annual_return']*100:.2f}%\n"
                f"Max Drawdown: {self.results['max_drawdown']*100:.2f}%\n"
                f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}\n"
                f"Trades: {self.results['num_trades']}\n"
                f"Win Rate: {self.results['win_rate']*100:.2f}%\n"
                f"Buy & Hold Return: {self.results['buy_hold_return']*100:.2f}%"
            )
            
            fig.text(0.02, 0.02, performance_text, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"回测图保存至: {save_path}")
            plt.show()
            logger.info("回测结果图绘制完成")
        except Exception as e:
            logger.error(f"绘图失败: {e}")
            raise e

# 针对单一股票运行回测
def run_backtest_for_stock(symbol, data_dir=None, strategy_dir=None, results_dir=None, initial_capital=None, buy_cash_ratio=None):
    """
    针对指定股票代码执行回测，使用相对路径加载数据，并将结果保存到指定目录
    
    参数:
    --------
    symbol: str
      股票代码
    data_dir: str, optional
      原始数据目录（默认：配置文件中指定）
    strategy_dir: str, optional
      策略数据目录（默认：配置文件中指定）
    results_dir: str, optional
      结果目录（默认：配置文件中指定）
    initial_capital: float, optional
      初始资金（默认：配置文件中指定）
    buy_cash_ratio: float, optional
      买入时使用的现金比例（默认：配置文件中指定）
    """
    project_root = get_project_root()
    config = load_config()
    if data_dir is None:
        data_dir = os.path.join(project_root, config.get("data_dir", "data/featured"))
    if strategy_dir is None:
        strategy_dir = os.path.join(project_root, config.get("strategy_dir", "data/strategy"))
    if results_dir is None:
        results_dir = os.path.join(project_root, config.get("results_dir", "results/backtest"))
    if initial_capital is None:
        initial_capital = config.get("initial_capital", 1000000)
    if buy_cash_ratio is None:
        buy_cash_ratio = config.get("buy_cash_ratio", 0.9)
    
    original_file = os.path.join(data_dir, f"{symbol}_1wk_cleaned_featured.csv")
    strategy_file = os.path.join(strategy_dir, f"{symbol}_1wk_cleaned_featured_standardized_strategy.csv")
    
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"开始回测 {symbol}，原始数据: {original_file}，策略数据: {strategy_file}")
    backtest = StrategyBacktester(initial_capital=initial_capital, buy_cash_ratio=buy_cash_ratio)
    
    try:
        backtest.load_data(original_file, strategy_file)
    except FileNotFoundError as e:
        logger.error(f"{symbol} 数据文件未找到: {e}")
        return None
    except Exception as e:
        logger.error(f"{symbol} 数据加载失败: {e}")
        return None
    
    results = backtest.run_backtest()
    
    plot_path = os.path.join(results_dir, f"{symbol}_backtest_results.png")
    backtest.plot_results(save_path=plot_path)
    
    results_csv = os.path.join(results_dir, f"{symbol}_backtest_data.csv")
    try:
        backtest.data.to_csv(results_csv, index=False)
        logger.info(f"详细回测数据已保存至: {results_csv}")
    except Exception as e:
        logger.error(f"保存回测数据失败: {e}")
    
    logger.info(f"=== {symbol} Backtest Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            msg = f"{key}: {value*100:.2f}%" if "rate" in key or "return" in key else f"{key}: {value:.4f}"
            logger.info(msg)
        else:
            logger.info(f"{key}: {value}")
    
    return results

def compare_strategies(all_results):
    """
    生成所有策略绩效指标的对比图表，并保存为PNG文件
    """
    metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']
    comparison_data = []
    
    for symbol, results in all_results.items():
        row = {'Symbol': symbol}
        for metric in metrics:
            value = results.get(metric, 0)
            if 'return' in metric or 'rate' in metric or 'drawdown' in metric:
                value *= 100
            row[metric] = value
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    compare_csv = os.path.join(get_project_root(), "results", "backtest", "strategy_comparison.csv")
    comparison_df.to_csv(compare_csv, index=False)
    logger.info(f"策略对比数据已保存至: {compare_csv}")
    
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        title = metric.replace('_', ' ').title()
        if 'return' in metric.lower() or 'rate' in metric.lower() or 'drawdown' in metric.lower():
            title += ' (%)'
        sorted_df = comparison_df.sort_values(by=metric)
        color = 'green' if 'return' in metric or 'ratio' in metric or 'rate' in metric else 'red'
        bars = plt.barh(sorted_df['Symbol'], sorted_df[metric], color=color)
        plt.title(title)
        plt.grid(True, axis='x')
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else 0
            plt.text(label_x_pos + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
    plt.tight_layout()
    compare_img = os.path.join(get_project_root(), "results", "backtest", "strategy_comparison.png")
    plt.savefig(compare_img, dpi=300, bbox_inches='tight')
    logger.info(f"策略对比图保存至: {compare_img}")
    plt.show()

def main():
    """
    主函数，针对配置文件中指定的股票代码执行回测，并生成对比图
    """
    project_root = get_project_root()
    config = load_config()
    data_dir = os.path.join(project_root, config.get("data_dir", "data/featured"))
    strategy_dir = os.path.join(project_root, config.get("strategy_dir", "data/strategy"))
    
    # 从配置文件中获取选中的股票代码
    selected_symbols = config.get("selected_symbols", [])
    if not selected_symbols:
        logger.error("配置文件中未指定 selected_symbols 或为空")
        return
    
    logger.info(f"将对以下股票进行回测: {selected_symbols}")
    
    all_results = {}
    for symbol in selected_symbols:
        logger.info(f"\n开始回测 {symbol}...")
        results = run_backtest_for_stock(symbol, data_dir=data_dir, strategy_dir=strategy_dir)
        if results:
            all_results[symbol] = results
    
    if all_results:
        compare_strategies(all_results)
        logger.info("回测完成")

if __name__ == "__main__":
    main()
