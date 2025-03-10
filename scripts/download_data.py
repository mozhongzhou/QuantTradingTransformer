import os
import json
import yfinance as yf
from datetime import datetime
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
    """配置日志记录器，确保无乱码"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "download_data.log")
    
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

# ================= 数据下载 =================
def download_stock_data(stock, start_date, end_date, interval, save_dir, overwrite, max_retries=5):
    """
    下载单只股票数据并保存到CSV文件
    :param stock: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param interval: 数据间隔（如 "1d", "1wk"）
    :param save_dir: 保存目录
    :param overwrite: 是否覆盖同名文件
    :param max_retries: 最大重试次数
    """
    try:
        save_path = os.path.join(save_dir, f"{stock}_{interval}.csv")
        
        if not overwrite and os.path.exists(save_path):
            logger.info(f"{save_path} 已存在，跳过下载")
            return
        
        for attempt in range(max_retries):
            logger.info(f"开始下载 {stock} 数据（{start_date} 至 {end_date}，间隔：{interval}），尝试次数：{attempt + 1}")
            
            data = yf.download(
                tickers=stock,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                logger.warning(f"{stock} 未找到数据，跳过保存")
                return
            
            # 检查数据是否包含指定的开始日期
            if start_date in data.index:
                data.to_csv(save_path, encoding="utf-8")
                logger.info(f"{stock} 数据已保存至 {save_path}")
                return
            else:
                logger.warning(f"{stock} 数据不包含开始日期 {start_date}，重试下载")
        
        # 如果多次重试后仍然不符合要求，保留最后一次下载的数据
        data.to_csv(save_path, encoding="utf-8")
        logger.warning(f"{stock} 数据多次重试后仍不包含开始日期 {start_date}，保留最后一次下载的数据")
        
    except Exception as e:
        logger.error(f"下载 {stock} 数据时出错: {str(e)}")

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
    config = load_config()
    if not config:
        return
    
    stocks = config.get("stocks", [])
    start_date = config.get("start_date", "2010-01-01")
    end_date = config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    interval = config.get("interval", "1d")  # 默认固定为日 K 线
    overwrite = config.get("overwrite", False)  # 是否覆盖同名文件，默认为 False
    
    save_dir = os.path.join(get_project_root(), "data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用 ThreadPoolExecutor 进行多线程下载
    with ThreadPoolExecutor(max_workers=500) as executor:
        futures = {executor.submit(download_stock_data, stock, start_date, end_date, interval, save_dir, overwrite): stock for stock in stocks}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                future.result()
                logger.info(f"{stock} 下载完成")
            except Exception as e:
                logger.error(f"{stock} 下载失败: {str(e)}")

if __name__ == "__main__":
    main()