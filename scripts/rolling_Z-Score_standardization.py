import os
import pandas as pd
import numpy as np
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
    log_file = os.path.join(log_dir, "standardize_data.log")
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

# ================= 数据标准化函数 =================
def standardize_stock_data(file_path, output_path, window_size=30):
    """
    对单个股票数据文件进行时间序列感知的 Z-score 标准化，并去除 NaN 行
    :param file_path: 特征文件路径
    :param output_path: 标准化后文件保存路径
    :param window_size: 滚动窗口大小，用于计算均值和标准差
    """
    try:
        logger.info(f"开始标准化文件: {file_path}")

        # 读取特征文件
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])  # 确保日期列为日期格式
        df = df.sort_values("Date")  # 按日期排序

        # 检查数据完整性
        feature_cols = [col for col in df.columns if col != "Date"]  # 排除 Date 列
        if df[feature_cols].isnull().any().any():
            logger.warning(f"文件 {file_path} 特征列中存在缺失值，将在后续步骤中移除")
        
        # 滚动标准化
        eps = 1e-8  # 避免标准差为0
        for col in feature_cols:
            logger.info(f"对列 {col} 进行滚动标准化，窗口大小为 {window_size}")
            df[f"{col}_rolling_mean"] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[f"{col}_rolling_std"] = df[col].rolling(window=window_size, min_periods=1).std()
            df[f"{col}_standardized"] = (df[col] - df[f"{col}_rolling_mean"]) / (df[f"{col}_rolling_std"] + eps)

        # 保留原始列和标准化列
        columns_to_keep = ["Date"] + feature_cols + [f"{col}_standardized" for col in feature_cols]
        standardized_df = df[columns_to_keep]

        # 直接去除包含 NaN 的行
        initial_rows = len(standardized_df)
        standardized_df = standardized_df.dropna()
        removed_rows = initial_rows - len(standardized_df)
        if removed_rows > 0:
            logger.info(f"文件 {file_path} 中移除了 {removed_rows} 行含有 NaN 的数据")

        # 保存标准化后的数据
        standardized_df.to_csv(output_path, index=False)
        logger.info(f"标准化后的数据已保存至: {output_path}")

    except Exception as e:
        logger.error(f"标准化文件 {file_path} 时出错: {str(e)}")

# ================= 批量标准化函数 =================
def standardize_all_stock_data(input_dir, output_dir, window_size=30):
    """
    标准化指定目录下的所有股票数据文件
    :param input_dir: 特征文件目录
    :param output_dir: 标准化后文件保存目录
    :param window_size: 滚动窗口大小，用于计算均值和标准差
    """
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for file_name in os.listdir(input_dir):
            if file_name.endswith("_cleaned.csv"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_standardized.csv")
                futures.append(executor.submit(standardize_stock_data, input_path, output_path, window_size))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"标准化过程中发生错误: {str(e)}")

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始数据标准化 ==========")
        input_dir = os.path.join(get_project_root(), "data", "cleaned")
        output_dir = os.path.join(get_project_root(), "data", "standardized")
        standardize_all_stock_data(input_dir, output_dir)
        logger.info("========== 数据标准化完成 ==========")
    except Exception as e:
        logger.error(f"数据标准化过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()