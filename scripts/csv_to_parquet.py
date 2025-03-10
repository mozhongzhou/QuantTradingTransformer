import os
import gc
import logging
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ================= 配置参数 =================
WINDOW_SIZE = 30  # 时间窗口长度 (30天)
PREDICT_STEPS = 1  # 预测未来步长
DATA_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
STANDARDIZED_COLUMNS = ['Open_standardized', 'High_standardized', 'Low_standardized', 'Close_standardized', 'Volume_standardized']
TARGET_COLUMN = 'Close_standardized'  # 预测目标列

# ================= 项目路径管理 =================
def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger(name: str) -> logging.Logger:
    """配置日志记录器"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件处理器
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, f"{name}.log"),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger("data_pipeline")

# ================= 数据转换核心逻辑 =================
class ParquetConverter:
    """CSV转Parquet并生成时间窗口的核心处理器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.raw_parquet_dir = os.path.join(output_dir, "raw")
        self.processed_dir = os.path.join(output_dir, "processed")
        
        os.makedirs(self.raw_parquet_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    @staticmethod
    def _optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        # 日期处理
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # 数值列类型优化
        float_cols = df.select_dtypes(include='float').columns
        df[float_cols] = df[float_cols].astype(np.float32)
        
        # 处理可能的无穷大值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df

    def _convert_single_file(self, csv_path: str) -> Optional[str]:
        """处理单个CSV文件"""
        try:
            # 读取并优化数据
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df = self._optimize_dataframe(df)
            
            # 生成唯一文件名
            file_id = os.path.splitext(os.path.basename(csv_path))[0]
            parquet_path = os.path.join(self.raw_parquet_dir, f"{file_id}.parquet")
            
            # 写入Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                parquet_path,
                compression='ZSTD',
                version='2.6',
                use_dictionary=True
            )
            logger.info(f"成功转换 {csv_path} 到 {parquet_path}")
            return parquet_path
            
        except Exception as e:
            logger.error(f"处理文件 {csv_path} 失败: {str(e)}")
            return None

    def convert_all_csvs(self) -> None:
        """批量转换CSV文件到Parquet格式"""
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = {}
            for fname in os.listdir(self.input_dir):
                if fname.endswith(".csv"):
                    csv_path = os.path.join(self.input_dir, fname)
                    futures[executor.submit(self._convert_single_file, csv_path)] = fname

            for future in as_completed(futures):
                fname = futures[future]
                try:
                    result = future.result()
                    if result:
                        logger.debug(f"处理完成: {fname} => {result}")
                except Exception as e:
                    logger.error(f"处理 {fname} 时发生未捕获错误: {str(e)}")

# ================= 时间窗口数据集 =================
class TimeSeriesDataset:
    """时间序列数据集（仅用于数据准备）"""
    
    def __init__(self, parquet_path: str):
        # 使用内存映射加速读取
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.data = self.parquet_file.read(columns=STANDARDIZED_COLUMNS + [TARGET_COLUMN, 'Date'])
        self.dates = self.data.column('Date').to_pandas()
        self.features = self.data.drop(['Date']).to_pandas().values.astype(np.float32)
        self.targets = self.data.column(TARGET_COLUMN).to_pandas().values.astype(np.float32)
        
        # 预计算有效窗口
        self.valid_indices = self._precompute_valid_windows()
        
    def _precompute_valid_windows(self) -> list:
        """预计算有效的窗口索引"""
        valid = []
        for i in range(len(self.features) - WINDOW_SIZE - PREDICT_STEPS + 1):
            end_idx = i + WINDOW_SIZE
            if not np.isnan(self.features[i:end_idx]).any():
                valid.append(i)
        return valid

    def save_processed_data(self, output_path: str) -> None:
        """保存处理后的数据"""
        processed_data = []
        for idx in self.valid_indices:
            end_idx = idx + WINDOW_SIZE
            target_idx = end_idx + PREDICT_STEPS - 1
            
            # 获取数据窗口
            features = self.features[idx:end_idx]
            target = self.targets[target_idx]
            date = self.dates.iloc[target_idx]
            
            # 保存为字典
            processed_data.append({
                'date': date,
                'features': features.tolist(),
                'target': target
            })
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(processed_data)
        df.to_parquet(output_path, compression='ZSTD')
        logger.info(f"处理后的数据已保存至: {output_path}")

# ================= 数据管道主控类 =================
class DataPipeline:
    """端到端数据管道"""
    
    def __init__(self):
        self.root_dir = get_project_root()
        self.data_dir = os.path.join(self.root_dir, "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        
        self.converter = ParquetConverter(self.raw_dir, self.processed_dir)

    def run_pipeline(self) -> None:
        """执行完整数据处理流程"""
        logger.info("启动数据处理管道")
        
        # 阶段1: 数据转换
        logger.info("开始CSV到Parquet转换")
        self.converter.convert_all_csvs()
        
        # 阶段2: 处理单个文件
        parquet_files = [
            os.path.join(self.converter.raw_parquet_dir, f)
            for f in os.listdir(self.converter.raw_parquet_dir)
            if f.endswith(".parquet")
        ]
        
        if not parquet_files:
            raise FileNotFoundError("未找到任何Parquet文件")
            
        # 使用第一个文件创建示例数据集
        dataset = TimeSeriesDataset(parquet_files[0])
        
        # 阶段3: 保存处理后的数据
        output_path = os.path.join(self.processed_dir, "processed_data.parquet")
        dataset.save_processed_data(output_path)
        
        logger.info("========== 数据处理成功完成 ==========")

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 启动数据处理系统 ==========")
        
        # 初始化并运行管道
        pipeline = DataPipeline()
        pipeline.run_pipeline()
        
    except Exception as e:
        logger.error(f"数据处理管道失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # 配置环境
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    main()