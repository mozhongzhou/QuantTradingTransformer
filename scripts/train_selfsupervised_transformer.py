# train_selfsupervised_transformer.py
# -*- coding: utf-8 -*-

import os
import json
import logging
import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger(log_filename="train_selfsupervised_transformer.log", log_level=logging.INFO):
    """
    设置日志记录器，输出到控制台和日志文件。
    参数:
        log_filename (str): 日志文件名
        log_level (int): 日志级别，默认 INFO
    """
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("train_selfsupervised_transformer")
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

logger = setup_logger()
logger.info("日志系统初始化成功")

# ================= 数据加载 =================
def load_csv_data(data_dir: str) -> pd.DataFrame:
    """
    从指定目录加载所有符合命名规则的 CSV 文件，并合并为一个 DataFrame。
    参数:
        data_dir (str): 存放 CSV 文件的目录。
    返回:
        pd.DataFrame: 合并后的数据。
    """
    data_frames = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_cleaned_featured_standardized.csv"):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            stock_code = filename.split("_")[0]
            df["Stock"] = stock_code
            data_frames.append(df)
    if not data_frames:
        raise ValueError("未在指定目录中找到符合条件的 CSV 文件。")
    return pd.concat(data_frames, ignore_index=True)

# ================= 数据集定义 =================
class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_length: int):
        if "Returns" not in df.columns:
            raise ValueError("数据中必须包含 'Returns' 列用于计算奖励。")
        self.raw_returns = df["Returns"].reset_index(drop=True)
        std_cols = [col for col in df.columns if col.endswith("_standardized") and col != "Returns_standardized"]
        if not std_cols:
            raise ValueError("未找到标准化特征列")
        self.input_data = df[std_cols].reset_index(drop=True)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.input_data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.input_data.iloc[idx: idx + self.seq_length].values
        target_return = self.raw_returns.iloc[idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target_return, dtype=torch.float32)

# ================= 数据集划分函数 =================
def split_dataset(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df

# ================= 位置编码 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ================= Transformer 模型定义 =================
class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, output_dim: int):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        encoded = self.encoder(x)
        last_output = encoded[-1]
        signal = self.head(last_output)
        signal = torch.tanh(signal)
        return signal.squeeze(-1)

# ================= 奖励函数 =================
def calculate_reward(target_returns: torch.Tensor, signals: torch.Tensor, risk_penalty: float) -> torch.Tensor:
    strategy_returns = signals * target_returns
    annual_return = strategy_returns.mean() * 252
    risk = torch.std(strategy_returns)
    reward = annual_return - risk_penalty * risk
    return reward

# ================= 训练逻辑 =================
def train_model(model: nn.Module, dataloader: DataLoader, config: dict, device: torch.device, phase="训练", optimizer=None):
    risk_penalty = config.get("risk_penalty", 0.5)
    if phase == "训练":
        if optimizer is None:
            raise ValueError("训练阶段需要传入优化器")
        model.train()
    else:
        model.eval()

    epoch_rewards = []
    annual_returns = []
    risks = []
    for sequences, target_returns in dataloader:
        sequences = sequences.to(device)
        target_returns = target_returns.to(device)
        if phase == "训练":
            optimizer.zero_grad()
        signals = model(sequences)
        strategy_returns = signals * target_returns
        annual_return = strategy_returns.mean() * 252
        risk = torch.std(strategy_returns)
        reward = annual_return - risk_penalty * risk
        if phase == "训练":
            loss = -reward
            loss.backward()
            optimizer.step()
        epoch_rewards.append(reward.item())
        annual_returns.append(annual_return.item())
        risks.append(risk.item())

    avg_reward = sum(epoch_rewards) / len(epoch_rewards)
    avg_annual_return = sum(annual_returns) / len(annual_returns)
    avg_risk = sum(risks) / len(risks)
    logger.info(f"{phase}阶段 - 平均奖励: {avg_reward:.4f}, 平均年化收益: {avg_annual_return:.4f}, 平均风险: {avg_risk:.4f}")
    return {"avg_reward": avg_reward, "avg_annual_return": avg_annual_return, "avg_risk": avg_risk}

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始训练自监督 Transformer 模型 ==========")
        project_root = get_project_root()
        data_dir = os.path.join(project_root, "data", "standardized")
        df = load_csv_data(data_dir)
        logger.info(f"数据加载完成，共 {len(df)} 条记录")

        config_path = os.path.join(project_root, "configs", "train_selfsupervised_transformer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"配置文件加载完成: {config}")

        if config.get("small_scale", False):
            max_samples = config.get("max_samples", 1000)
            df = df.head(max_samples)
            logger.info(f"小规模试跑：使用前 {max_samples} 条记录")

        train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        logger.info(f"数据集划分完成：训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条, 测试集 {len(test_df)} 条")

        seq_length = config.get("seq_length", 30)
        batch_size = config.get("batch_size", 64)
        train_dataset = TimeSeriesDataset(train_df, seq_length=seq_length)
        val_dataset = TimeSeriesDataset(val_df, seq_length=seq_length)
        test_dataset = TimeSeriesDataset(test_df, seq_length=seq_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

        input_dim = train_dataset.input_data.shape[1]
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            output_dim=config["output_dim"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("模型初始化完成")

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        model_save_dir = os.path.join(project_root, "models")
        os.makedirs(model_save_dir, exist_ok=True)

        best_val_reward = -float("inf")
        for epoch in range(config["num_epochs"]):
            logger.info(f"===== Epoch {epoch+1}/{config['num_epochs']} =====")
            train_metrics = train_model(model, train_loader, config, device, phase="训练", optimizer=optimizer)
            val_metrics = train_model(model, val_loader, config, device, phase="验证")
            if val_metrics["avg_reward"] > best_val_reward:
                best_val_reward = val_metrics["avg_reward"]
                model_name = f"transformer_model_best.pth"
                model_save_path = os.path.join(model_save_dir, model_name)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"验证集奖励提升，模型已保存至: {model_save_path}")

        logger.info("========== 模型在测试集上评估 ==========")
        test_metrics = train_model(model, test_loader, config, device, phase="测试")
        logger.info(f"测试集 - 平均奖励: {test_metrics['avg_reward']:.4f}, 平均年化收益: {test_metrics['avg_annual_return']:.4f}, 平均风险: {test_metrics['avg_risk']:.4f}")
        logger.info("========== 训练完成 ==========")

    except Exception as e:
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()