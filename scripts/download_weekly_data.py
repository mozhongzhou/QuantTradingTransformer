# scripts/download_weekly_data.py
import os
import json
import yfinance as yf
from datetime import datetime
import logging

# ================= 路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_raw_data_path(stock_code=""):
    """获取原始数据存储路径"""
    raw_dir = os.path.join(get_project_root(), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return os.path.join(raw_dir, f"{stock_code}_weekly.csv" if stock_code else "")

# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器，确保无乱码"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "download_weekly_data.log")
    
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
def download_stock_data(stock, start_date, end_date, interval, save_dir):
    """
    下载单只股票周 K 线数据并保存到CSV文件
    :param stock: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param interval: 数据间隔（固定为 1wk）
    :param save_dir: 保存目录
    """
    try:
        logger.info(f"开始下载 {stock} 周 K 线数据（{start_date} 至 {end_date}）")
        
        data = yf.download(
            tickers=stock,
            start=start_date,
            end=end_date,
            interval=interval,  # 固定为 "1wk"
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            logger.warning(f"{stock} 未找到周 K 线数据，跳过保存")
            return
        
        save_path = os.path.join(save_dir, f"{stock}_weekly.csv")
        data.to_csv(save_path, encoding="utf-8")
        logger.info(f"{stock} 周 K 线数据已保存至 {save_path}")
        
    except Exception as e:
        logger.error(f"下载 {stock} 周 K 线数据时出错: {str(e)}")

# ================= 配置文件读取 =================
def load_config():
    """加载配置文件"""
    config_path = os.path.join(get_project_root(), "configs", "stocks.json")
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
    interval = "1wk"  # 固定为周 K 线
    
    save_dir = os.path.join(get_project_root(), "data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    for stock in stocks:
        download_stock_data(stock, start_date, end_date, interval, save_dir)

if __name__ == "__main__":
    main()