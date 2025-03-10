import os
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ================= 日志配置 =================
def setup_logger() -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger("csv_to_parquet")
    logger.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= CSV 转 Parquet =================
def csv_to_parquet(input_dir: str, output_dir: str) -> None:
    """
    将指定目录下的所有CSV文件转换为Parquet文件。
    
    参数:
        input_dir (str): 输入目录，包含CSV文件。
        output_dir (str): 输出目录，用于保存Parquet文件。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有CSV文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_dir, filename)
            parquet_path = os.path.join(output_dir, filename.replace(".csv", ".parquet"))

            try:
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                
                # 将DataFrame写入Parquet文件
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_path, compression='ZSTD')
                
                logger.info(f"成功转换: {csv_path} -> {parquet_path}")
            except Exception as e:
                logger.error(f"转换失败: {csv_path} -> {str(e)}")

if __name__ == "__main__":
    # 获取项目根目录
    project_root = get_project_root()

    # 输入和输出目录（相对路径）
    input_dir = os.path.join(project_root, "data", "standardized")  # 存放CSV文件的目录
    output_dir = os.path.join(project_root, "data", "parquet")  # 存放Parquet文件的目录

    # 执行转换
    logger.info("========== 开始CSV转Parquet ==========")
    csv_to_parquet(input_dir, output_dir)
    logger.info("========== 转换完成 ==========")