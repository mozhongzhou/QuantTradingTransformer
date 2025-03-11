# test_models.py
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger(log_filename="test_models.log", log_level=logging.INFO):
    """
    设置日志记录器，输出到控制台和日志文件。
    参数:
        log_filename (str): 日志文件名
        log_level (int): 日志级别，默认 INFO
    """
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("test_models")
    logger.setLevel(log_level)
    logger.handlers.clear()
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

# 设置日志
logger = setup_logger()
logger.info("日志系统初始化成功")

# ================= 导入其他模块 =================
try:
    from train_selfsupervised_transformer import (
        load_csv_data, TimeSeriesDataset, TransformerModel, time_based_split
    )
    logger.info("成功导入训练模块")
except ImportError as e:
    logger.error(f"导入训练模块失败: {str(e)}")
    raise

# ================= 配置管理 =================
def load_config(config_path=None):
    """
    加载测试配置，优先使用指定的配置文件路径，否则使用默认配置文件
    
    参数:
        config_path (str): 配置文件路径，可以是相对路径
        
    返回:
        dict: 配置字典
    """
    project_root = get_project_root()
    
    if config_path is None:
        config_path = os.path.join(project_root, "configs", "test_model_config.json")
    elif not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
        
        default_config = {
            "model_path": os.path.join("models", "transformer_model_best.pth"),
            "data_dir": os.path.join("data", "standardized"),
            "output_dir": os.path.join("signals"),
            "batch_size": 128,
            "use_gpu": True,
            "save_return_analysis": True
        }
        
        config_dir = os.path.dirname(config_path)
        os.makedirs(config_dir, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        
        logger.info(f"已创建默认配置文件: {config_path}")
        return default_config
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    logger.info(f"成功加载配置文件: {config_path}")
    
    required_fields = ["stock_codes", "model_path"]
    for field in required_fields:
        if field not in config:
            logger.error(f"缺失必要配置项: {field}")
            raise ValueError(f"Invalid config: missing {field}")
    
    if not isinstance(config["stock_codes"], list):
        logger.error("stock_codes必须为列表格式")
        raise TypeError("stock_codes should be list")
    
    return config

# ================= 模型加载 =================
def load_model(model_path, data_dir):
    """
    智能加载模型 - 自动检测模型文件类型并提取参数
    
    参数:
        model_path (str): 模型文件路径，可以是相对路径
        data_dir (str): 数据目录，用于确定输入特征维度
        
    返回:
        model: 加载好的模型对象
        config: 模型配置
    """
    project_root = get_project_root()
    
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model_data = torch.load(model_path, map_location=device)
    
    sample_df = load_csv_data(data_dir)
    std_cols = [col for col in sample_df.columns if col.endswith("_standardized") and col != "Returns_standardized"]
    input_dim = len(std_cols)
    
    if isinstance(model_data, dict) and 'model_state_dict' in model_data:
        logger.info("检测到检查点文件，提取模型配置...")
        config = model_data.get('config', {})
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            output_dim=config.get("output_dim", 1)
        )
        model.load_state_dict(model_data['model_state_dict'])
    else:
        logger.info("检测到模型状态字典文件，尝试提取配置...")
        
        model_dir = os.path.dirname(model_path)
        checkpoint_path = os.path.join(model_dir, "checkpoints", "latest_checkpoint.pt")
        
        if os.path.exists(checkpoint_path):
            logger.info(f"从检查点文件提取配置: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get('config', {})
        else:
            logger.info("未找到检查点，使用默认配置文件")
            config_path = os.path.join(project_root, "configs", "train_selfsupervised_transformer_config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            output_dim=config.get("output_dim", 1)
        )
        model.load_state_dict(model_data)
    
    model.to(device)
    logger.info(f"模型加载成功，输入维度: {input_dim}")
    
    return model, config

def generate_and_save_signals(config=None):
    """
    评估模型并生成交易信号
    
    参数:
        config (dict): 测试配置，如果为None则加载默认配置
        
    返回:
        str: 生成的信号文件路径，如果生成失败则返回 None
    """
    project_root = get_project_root()
    
    if config is None:
        config = load_config()
    
    model_path = config.get("model_path")
    data_dir = config.get("data_dir")
    output_dir = config.get("output_dir")
    batch_size = config.get("batch_size", 128)
    target_stocks = config.get("stock_codes", [])

    model_path = os.path.join(project_root, model_path) if not os.path.isabs(model_path) else model_path
    data_dir = os.path.join(project_root, data_dir) if not os.path.isabs(data_dir) else data_dir
    output_dir = os.path.join(project_root, output_dir) if not os.path.isabs(output_dir) else output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu")
    logger.info(f"使用设备: {device}")

    if not isinstance(target_stocks, list) or len(target_stocks) == 0:
        logger.error("配置错误: stock_codes 必须为非空列表")
        return None
    if len(target_stocks) > 5:
        logger.warning("建议同时测试的股票不超过5只，当前数量: %d", len(target_stocks))

    try:
        logger.info(f"加载模型: {model_path}")
        model, model_config = load_model(model_path, data_dir)

        logger.info(f"加载数据: {data_dir}")
        full_df = load_csv_data(data_dir)
        filtered_df = full_df[full_df["Stock"].isin(target_stocks)].copy()
        
        if filtered_df.empty:
            missing_stocks = set(target_stocks) - set(full_df["Stock"].unique())
            logger.error(f"以下股票不存在于数据中: {', '.join(missing_stocks)}")
            return None

        all_signals = []
        seq_length = model_config.get("seq_length", 30)
        
        for stock_code, stock_df in filtered_df.groupby("Stock"):
            logger.info(f"开始处理: {stock_code}")
            
            dataset = TimeSeriesDataset(stock_df, seq_length=seq_length)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            signals, dates, returns = [], [], []
            model.eval()
            with torch.no_grad():
                for batch_idx, (seq, ret) in enumerate(dataloader):
                    seq = seq.to(device)
                    pred = model(seq).cpu().numpy().flatten()
                    
                    signals.extend(pred)
                    returns.extend(ret.numpy().flatten())
                    
                    for i in range(len(pred)):
                        global_idx = batch_idx * batch_size + i
                        if global_idx < len(dataset.valid_sequences):
                            _, start_idx, _ = dataset.valid_sequences[global_idx]
                            date_idx = start_idx + seq_length
                            if date_idx < len(stock_df):
                                dates.append(stock_df.iloc[date_idx]["Date"])

            if len(dates) != len(signals):
                logger.error(f"{stock_code} 日期与信号数量不匹配，跳过该股票")
                continue
                
            stock_signal_df = pd.DataFrame({
                "Date": dates,
                "Stock": stock_code,
                "Signal": signals,
                "Return": returns
            })
            all_signals.append(stock_signal_df)
            logger.info(f"{stock_code} 生成 {len(signals)} 条信号")

        if not all_signals:
            logger.error("所有股票处理失败")
            return None
            
        signal_df = pd.concat(all_signals).sort_values(["Stock", "Date"])
        signal_df["Strategy_Return"] = signal_df["Signal"] * signal_df["Return"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"signals_{timestamp}.csv")
        signal_df.to_csv(output_path, index=False)
        logger.info(f"信号文件已保存至: {output_path}")

        if config.get("save_return_analysis", True):
            analysis_df = signal_df.groupby("Stock").apply(
                lambda g: pd.Series({
                    "Avg_Signal": g.Signal.mean(),
                    "Win_Rate": (g.Strategy_Return > 0).mean(),
                    "Sharpe": g.Strategy_Return.mean() / g.Strategy_Return.std() * np.sqrt(252),
                    "Max_Drawdown": (g.Strategy_Return.cumsum().expanding().max() - g.Strategy_Return.cumsum()).max()
                })
            ).reset_index()
            
            analysis_path = os.path.join(output_dir, f"performance_{timestamp}.csv")
            analysis_df.to_csv(analysis_path, index=False)
            logger.info(f"绩效分析已保存至: {analysis_path}")

        return output_path

    except Exception as e:
        logger.error(f"信号生成失败: {str(e)}", exc_info=True)
        return None

# ================= 主函数 =================
def main():
    """主函数，处理命令行参数并执行信号生成"""
    
    parser = argparse.ArgumentParser(description="评估模型并生成交易信号")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info("========== 开始生成交易信号 ==========")
    logger.info(f"使用配置: {json.dumps(config, indent=2)}")
    
    try:
        signal_path = generate_and_save_signals(config)
        if signal_path:
            logger.info("信号生成完成！")
        else:
            logger.error("信号生成失败！")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("========== 信号生成过程结束 ==========")

if __name__ == "__main__":
    main()