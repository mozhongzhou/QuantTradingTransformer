import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

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
    log_file = os.path.join(log_dir, "standardize_data.log")

    # 配置日志格式和编码
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
def standardize_stock_data(file_path, output_path):
    """
    对单个股票数据文件进行 Z-score 标准化
    :param file_path: 特征文件路径
    :param output_path: 标准化后文件保存路径
    """
    try:
        logger.info(f"开始标准化文件: {file_path}")

        # 读取特征文件
        df = pd.read_csv(file_path)
        feature_cols = [col for col in df.columns if col != "Date"]  # 排除 Date 列
        features = df[feature_cols].values

        # Z-score 标准化
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(features)

        # 将标准化后的数据放回 DataFrame
        standardized_df = pd.DataFrame(standardized_features, columns=feature_cols)
        standardized_df["Date"] = df["Date"]  # 保留原始 Date 列

        # 确保 Date 列是第一列
        cols = ["Date"] + feature_cols
        standardized_df = standardized_df[cols]

        # 保存标准化后的数据
        standardized_df.to_csv(output_path, index=False)
        logger.info(f"标准化后的数据已保存至: {output_path}")

    except Exception as e:
        logger.error(f"标准化文件 {file_path} 时出错: {str(e)}")

# ================= 批量标准化函数 =================
def standardize_all_stock_data(input_dir, output_dir):
    """
    标准化指定目录下的所有股票数据文件
    :param input_dir: 特征文件目录
    :param output_dir: 标准化后文件保存目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith("_featured.csv"):
            # 构造输入和输出文件路径
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_standardized.csv")

            # 标准化数据并保存
            standardize_stock_data(input_path, output_path)

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始数据标准化 ==========")

        # 配置路径
        input_dir = os.path.join(get_project_root(), "data", "featured")  # 特征数据目录
        output_dir = os.path.join(get_project_root(), "data", "standardized")  # 标准化后数据目录

        # 标准化所有股票数据
        standardize_all_stock_data(input_dir, output_dir)

        logger.info("========== 数据标准化完成 ==========")

    except Exception as e:
        logger.error(f"数据标准化过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()