import os
import json
import yfinance as yf
from datetime import datetime
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_raw_data_path(stock_code="", interval=""):
    """获取原始数据存储路径"""
    raw_dir = os.path.join(get_project_root(), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return os.path.join(raw_dir, f"{stock_code}_{interval}.csv" if stock_code else "")

# ================= 日志配置 =================
def setup_logger():
    """简化版日志配置"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(log_dir, "download_and_clean_data.log")

    # 配置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M"  # 仅显示年月日时分
    )

    # 获取或创建日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 添加处理器（避免重复添加）
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

# ================= 数据下载与清洗函数 =================
def download_and_clean_stock_data(stock, start_date, end_date, interval, output_dir, overwrite):
    """
    下载并清洗单只股票数据，保存为清洗后的文件
    :param stock: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param interval: 数据间隔（如 "1d", "1wk"）
    :param output_dir: 清洗后文件保存目录
    :param overwrite: 是否覆盖同名文件
    """
    try:
        # 构造输出文件路径
        output_path = os.path.join(output_dir, f"{stock}_{interval}_cleaned.csv")
        
        # 如果文件已存在且不覆盖，则跳过
        if not overwrite and os.path.exists(output_path):
            logger.info(f"{output_path} 已存在，跳过下载和清洗")
            return
        
        # 下载数据
        logger.info(f"开始下载 {stock} 数据（{start_date} 至 {end_date}，间隔：{interval}）")
        data = yf.download(
            tickers=stock,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        # 检查数据是否为空
        if data.empty:
            logger.warning(f"{stock} 未找到数据，跳过保存")
            return
        
        # 清洗数据
        logger.info(f"开始清洗 {stock} 数据")
        data = data.reset_index()  # 将日期列从索引转换为普通列
        data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]  # 设置正确的列名

        # 检查数据类型
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")  # 将日期列转换为日期类型
        for col in ["Open", "Close", "High", "Low", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")  # 将数值列转换为浮点数

        # 处理缺失值
        if data.isnull().sum().sum() > 0:
            logger.warning(f"{stock} 数据中存在缺失值，已删除相关行")
        data.dropna(inplace=True)  # 删除包含缺失值的行

        # 保存清洗后的数据
        data.to_csv(output_path, index=False)
        logger.info(f"{stock} 数据已下载并清洗，保存至: {output_path}")

    except Exception as e:
        logger.error(f"下载或清洗 {stock} 数据时出错: {str(e)}")

# ================= 批量下载与清洗函数 =================
def download_and_clean_all_stock_data(stocks, start_date, end_date, interval, output_dir, overwrite):
    """
    下载并清洗所有股票数据
    :param stocks: 股票代码列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param interval: 数据间隔（如 "1d", "1wk"）
    :param output_dir: 清洗后文件保存目录
    :param overwrite: 是否覆盖同名文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for stock in stocks:
            futures.append(executor.submit(
                download_and_clean_stock_data,
                stock, start_date, end_date, interval, output_dir, overwrite
            ))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"处理过程中发生错误: {str(e)}")

# ================= 配置文件读取 =================
def load_config():
    """加载配置文件"""
    config_path = os.path.join(get_project_root(), "configs", "download_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"配置文件格式错误: {config_path}")
        return None

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始数据下载与清洗 ==========")

        # 加载配置文件
        config = load_config()
        if not config:
            return
        
        # 获取配置参数
        stocks = config.get("stocks", [])
        start_date = config.get("start_date", "2010-01-04")
        end_date = config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        interval = config.get("interval", "1d")
        overwrite = config.get("overwrite", False)

        # 配置输出目录
        output_dir = os.path.join(get_project_root(), "data", "cleaned")
        os.makedirs(output_dir, exist_ok=True)

        # 下载并清洗所有股票数据
        download_and_clean_all_stock_data(stocks, start_date, end_date, interval, output_dir, overwrite)

        logger.info("========== 数据下载与清洗完成 ==========")

    except Exception as e:
        logger.error(f"数据下载与清洗过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()