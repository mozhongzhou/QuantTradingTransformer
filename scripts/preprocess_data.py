# scripts/preprocess_data.py
import os
import numpy as np
import pandas as pd
import logging
import json

# ================= 路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_cleaned_data_path(stock_code="", granularity=""):
    """获取清洗后数据存储路径，支持不同时间粒度"""
    cleaned_dir = os.path.join(get_project_root(), "data", "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    if not granularity:
        raise ValueError("必须指定时间粒度（daily, weekly, yearly）")
    return os.path.join(cleaned_dir, f"{stock_code}_{granularity}_cleaned.csv" if stock_code else "")

def get_sequences_path(stock_code="", granularity=""):
    """获取序列数据存储路径，支持不同时间粒度"""
    sequences_dir = os.path.join(get_project_root(), "data", "sequences")
    os.makedirs(sequences_dir, exist_ok=True)
    if not granularity:
        raise ValueError("必须指定时间粒度（daily, weekly, yearly）")
    return os.path.join(sequences_dir, f"{stock_code}_{granularity}_sequences.npz" if stock_code else "")

# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "preprocess_data.log")
    
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

# ================= 配置文件读取 =================
def load_config():
    """加载配置文件"""
    config_path = os.path.join(get_project_root(), "configs", "preprocess_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"配置文件格式错误: {config_path}")
        return None

# ================= 序列生成 =================
def create_sequences(ticker, granularity, seq_len=20, periods_ahead=1):
    """为单只股票生成序列数据，支持不同粒度（含 Z-Score 标准化）"""
    file_path = get_cleaned_data_path(ticker, granularity)
    if not os.path.exists(file_path):
        logger.error(f"清洗文件未找到: {file_path}")
        return
    
    try:
        # 读取数据
        df = pd.read_csv(file_path, parse_dates=["Date"], encoding="utf-8")
        features = ["Open", "High", "Low", "Close", "Volume"]
        
        # 检查必要列
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            logger.error(f"{file_path} 缺少必要列: {missing_cols}")
            return
        
        # 检查数据完整性
        if df.empty:
            logger.warning(f"{file_path} 数据为空，跳过处理")
            return
        if df[features].isnull().any().any():
            logger.warning(f"{file_path} 包含缺失值，将填充为 0")
            df[features] = df[features].fillna(0)
        
        # 提取特征数据并标准化
        data = df[features].values
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        # 避免除以 0
        stds[stds == 0] = 1.0  # 如果标准差为 0，设为 1
        standardized_data = (data - means) / stds
        
        # 生成标签：未来 periods_ahead 周期的收盘价是否上涨
        df["future_close"] = df["Close"].shift(-periods_ahead)
        labels = (df["future_close"] > df["Close"]).astype(int).values
        
        # 生成序列和目标
        sequences = []
        targets = []
        max_idx = len(data) - seq_len - periods_ahead + 1
        if max_idx <= 0:
            logger.error(f"{ticker} 数据量不足（{len(data)} 行），无法生成序列（需要至少 {seq_len + periods_ahead} 行）")
            return
        
        for i in range(max_idx):
            seq = standardized_data[i:i+seq_len]
            target = labels[i+seq_len-1]
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)  # (num_samples, seq_len, num_features)
        targets = np.array(targets)      # (num_samples,)
        
        # 保存结果
        output_file = get_sequences_path(ticker, granularity)
        np.savez(output_file, sequences=sequences, targets=targets, means=means, stds=stds)
        logger.info(f"序列已保存至: {output_file}, 形状: {sequences.shape}, 标签分布: {np.mean(targets):.2f}")
        
    except Exception as e:
        logger.error(f"处理 {ticker}（粒度: {granularity}）时出错: {str(e)}")

# ================= 主函数 =================
def main():
    config = load_config()
    if not config:
        return
    
    # 从配置中获取参数
    granularities = config.get("granularities", ["daily", "weekly", "yearly"])
    stocks = config.get("stocks", [])
    seq_len = config.get("seq_len", 20)
    periods_ahead = config.get("periods_ahead", 1)
    
    for granularity in granularities:
        cleaned_dir = get_cleaned_data_path("", granularity)
        cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith(f"_{granularity}_cleaned.csv")]
        
        if not cleaned_files:
            logger.warning(f"未找到任何 _{granularity}_cleaned.csv 文件在: {cleaned_dir}")
            continue
        
        for file in cleaned_files:
            ticker = file.split(f"_{granularity}_cleaned.csv")[0]
            logger.info(f"开始为 {ticker} 生成 {granularity} 序列（seq_len={seq_len}, periods_ahead={periods_ahead}）")
            create_sequences(ticker, granularity, seq_len, periods_ahead)

if __name__ == "__main__":
    main()