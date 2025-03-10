# feature_engineering_for_daily_data.py
import os
import pandas as pd
import logging
from ta.trend import MACD, SMAIndicator  # 使用 ta 库计算技术指标
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器，确保日志文件存储到 logs/ 目录"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "feature_engineering.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# ================= 特征工程函数 =================
def add_features_to_stock_data(file_path, output_path):
    """
    为单个股票数据文件添加技术特征
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

        # 2. 添加移动平均线 (MA5, MA20, MA50, MA200)
        for window in [5, 20, 50, 200]:
            df[f"MA{window}"] = SMAIndicator(df["Close"], window=window).sma_indicator()

        # 3. 添加相对强弱指数 (RSI)
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

        # 4. 添加 MACD
        macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"] = macd.macd_diff()

        # 5. 添加布林带 (Bollinger Bands)
        bb = BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_Band_Upper"] = bb.bollinger_hband()
        df["BB_Band_Middle"] = bb.bollinger_mavg()
        df["BB_Band_Lower"] = bb.bollinger_lband()

        # 6. 添加波动率 (ATR)
        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
        df["ATR"] = atr.average_true_range()

        # 7. 添加成交量加权平均价格 (VWAP)
        vwap = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"])
        df["VWAP"] = vwap.volume_weighted_average_price()

        # 8. 添加 OBV (On-Balance Volume)
        obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
        df["OBV"] = obv.on_balance_volume()

        # 9. 添加波动率 (Volatility)
        df["Volatility"] = df["Returns"].rolling(window=20).std()

        # 处理新特征中的缺失值
        df.fillna(method="ffill", inplace=True)  # 前向填充
        df.fillna(0, inplace=True)  # 剩余缺失值填充为 0

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
    os.makedirs(output_dir, exist_ok=True)

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=50) as executor:  # 调整线程池大小
        futures = []
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_featured.csv")
                futures.append(executor.submit(add_features_to_stock_data, input_path, output_path))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"处理过程中发生错误: {str(e)}")

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始特征工程 ==========")

        input_dir = os.path.join(get_project_root(), "data", "cleaned")
        output_dir = os.path.join(get_project_root(), "data", "featured")

        process_all_stock_data(input_dir, output_dir)

        logger.info("========== 特征工程完成 ==========")

    except Exception as e:
        logger.error(f"特征工程过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()