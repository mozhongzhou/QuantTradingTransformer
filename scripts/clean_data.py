import os
import pandas as pd
import logging
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

    # 日志文件路径
    log_file = os.path.join(log_dir, "clean_data.log")

    # 配置日志格式和编码
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 删除已有处理器避免重复
    if logger.handlers:
        logger.handlers = []

    # 自定义日志格式（时间仅显示年月日时分）
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M"  # 仅显示年月日时分
    )

    # 文件处理器（UTF-8编码）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
# ================= 数据清洗函数 =================
def clean_stock_data(file_path, output_path):
    """
    清洗单个股票数据文件，使其符合标准格式
    :param file_path: 原始文件路径
    :param output_path: 清洗后文件保存路径
    """
    try:
        logger.info(f"开始清洗文件: {file_path}")

        # 读取原始文件，跳过前3行
        df = pd.read_csv(file_path, skiprows=3, header=None)

        # 检查列数是否为6
        if df.shape[1] != 6:
            logger.warning(f"文件 {file_path} 列数不为6，已跳过")
            return

        # 设置正确的列名
        df.columns = ["Date", "Open", "Close", "High", "Low", "Volume"]

        # 检查数据类型
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")  # 将日期列转换为日期类型
        for col in ["Open", "Close", "High", "Low", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # 将数值列转换为浮点数

        # 处理缺失值
        if df.isnull().sum().sum() > 0:
            logger.warning(f"文件 {file_path} 中存在缺失值，已删除相关行")
        df.dropna(inplace=True)  # 删除包含缺失值的行

        # 保存清洗后的数据
        df.to_csv(output_path, index=False)
        logger.info(f"清洗后的数据已保存至: {output_path}")

    except Exception as e:
        logger.error(f"清洗文件 {file_path} 时出错: {str(e)}")

# ================= 批量清洗函数 =================
def clean_all_stock_data(input_dir, output_dir):
    """
    清洗指定目录下的所有股票数据文件
    :param input_dir: 原始文件目录
    :param output_dir: 清洗后文件保存目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 使用 ThreadPoolExecutor 进行多线程清洗
    with ThreadPoolExecutor(max_workers=500) as executor:
        futures = []
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                # 构造输入和输出文件路径
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_cleaned.csv")
                futures.append(executor.submit(clean_stock_data, input_path, output_path))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"清洗过程中发生错误: {str(e)}")

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始数据清洗 ==========")

        # 配置路径
        input_dir = os.path.join(get_project_root(), "data", "raw")  # 原始数据目录
        output_dir = os.path.join(get_project_root(), "data", "cleaned")  # 清洗后数据目录

        # 清洗所有股票数据
        clean_all_stock_data(input_dir, output_dir)

        logger.info("========== 数据清洗完成 ==========")

    except Exception as e:
        logger.error(f"数据清洗过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()