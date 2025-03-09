import os
import pandas as pd
import numpy as np
import logging
from ta.trend import MACD, SMAIndicator  # 使用 ta 库计算技术指标
from ta.momentum import RSIIndicator

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器，确保日志文件存储到 logs/ 目录"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(log_dir, "feature_engineering.log")

    # 配置日志格式和编码
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # 确保日志文件使用 UTF-8 编码
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# ================= 特征工程函数 =================
def add_features_to_stock_data(file_path, output_path):
    """
    为单个股票数据文件添加特征
    :param file_path: 清洗后的文件路径
    :param output_path: 添加特征后的文件保存路径
    """
    try:
        logger.info(f"开始为文件添加特征: {file_path}")

        # 读取清洗后的 CSV 文件
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])  # 确保日期列为日期格式

        # 检查数据完整性
        required_columns = ["Date", "Open", "Close", "High", "Low", "Volume"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("数据缺少必要的列")

        # 1. 添加简单收益率 (Returns)
        df["Returns"] = df["Close"].pct_change()

        # 2. 添加移动平均线 (MA5, MA20)
        df["MA5"] = SMAIndicator(df["Close"], window=5).sma_indicator()  # 5周均线
        df["MA20"] = SMAIndicator(df["Close"], window=20).sma_indicator()  # 20周均线

        # 3. 添加相对强弱指数 (RSI)
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()  # 14周 RSI

        # 4. 添加 MACD
        macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()  # MACD 线
        df["MACD_Signal"] = macd.macd_signal()  # 信号线
        df["MACD_Diff"] = macd.macd_diff()  # MACD 差值

        # 5. 添加波动率 (Volatility)
        df["Volatility"] = df["Returns"].rolling(window=20).std()  # 20周收益率标准差

        # 处理新特征中的缺失值（由于滚动计算导致的 NaN）
        if df.isnull().sum().sum() > 0:
            logger.warning(f"文件 {file_path} 中新增特征存在缺失值，已填充为 0")
            df.fillna(0, inplace=True)  # 可以根据需求改为其他填充方法，如前向填充

        # 保存添加特征后的数据
        df.to_csv(output_path, index=False)
        logger.info(f"特征添加完成，数据已保存至: {output_path}")

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")

# ================= 批量特征工程函数 =================
def process_all_stock_data(input_dir, output_dir):
    """
    为指定目录下的所有股票数据文件添加特征
    :param input_dir: 清洗后的文件目录
    :param output_dir: 添加特征后的文件保存目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            # 构造输入和输出文件路径
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_featured.csv")

            # 为数据添加特征并保存
            add_features_to_stock_data(input_path, output_path)

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始特征工程 ==========")

        # 配置路径
        input_dir = os.path.join(get_project_root(), "data", "cleaned")  # 清洗后的数据目录
        output_dir = os.path.join(get_project_root(), "data", "featured")  # 特征工程后的数据目录

        # 为所有股票数据添加特征
        process_all_stock_data(input_dir, output_dir)

        logger.info("========== 特征工程完成 ==========")

    except Exception as e:
        logger.error(f"特征工程过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()