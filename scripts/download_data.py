import os
import json
import yfinance as yf
from datetime import datetime
import logging

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
    """配置带运行分隔符的日志系统"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 删除已有处理器避免重复
    if logger.handlers:
        logger.handlers = []

    # 自定义格式（年月日时分）
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M"  # 仅显示到分钟
    )

    # 文件处理器（UTF-8编码）
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "download_data.log"),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

# ================= 数据下载 =================
def download_stock_data(stock, start_date, end_date, interval, save_dir, overwrite):
    """
    下载单只股票数据并保存到CSV文件
    :param stock: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param interval: 数据间隔（如 "1d", "1wk"）
    :param save_dir: 保存目录
    :param overwrite: 是否覆盖同名文件
    """
    try:
        save_path = os.path.join(save_dir, f"{stock}_{interval}.csv")
        
        if not overwrite and os.path.exists(save_path):
            logger.info(f"{save_path} 已存在，跳过下载")
            return
        
        logger.info(f"开始下载 {stock} 数据（{start_date} 至 {end_date}，间隔：{interval}）")
        
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
        
        data.to_csv(save_path, encoding="utf-8")
        logger.info(f"{stock} 数据已保存至 {save_path}")
        
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
    interval = config.get("interval", "1d")
    overwrite = config.get("overwrite", False)
    
    save_dir = os.path.join(get_project_root(), "data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    
    # 单线程顺序执行
    for stock in stocks:
        try:
            download_stock_data(stock, start_date, end_date, interval, save_dir, overwrite)
            logger.info(f"{stock} 下载完成")
        except Exception as e:
            logger.error(f"{stock} 下载失败: {str(e)}")

if __name__ == "__main__":
    main()