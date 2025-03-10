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

# ================= 日志配置 =================
def setup_logger() -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger("train_selfsupervised_transformer")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()


# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            # 可选：添加股票代码作为标识（假设文件名格式为 '股票代码_1d_cleaned_featured_standardized.csv'）
            stock_code = filename.split("_")[0]
            df["Stock"] = stock_code
            data_frames.append(df)
    if not data_frames:
        raise ValueError("未在指定目录中找到符合条件的 CSV 文件。")
    # 假设所有文件的数据结构一致，并且数据已经按照日期升序排列
    return pd.concat(data_frames, ignore_index=True)

# ================= 数据集定义 =================
class TimeSeriesDataset(Dataset):
    """
    将时间序列数据切分为滑动窗口序列，每个样本包含固定长度的历史标准化特征和对应下一天的原始收益(Returns)作为目标。
    """
    def __init__(self, df: pd.DataFrame, seq_length: int):
        # 检查原始收益列是否存在
        if "Returns" not in df.columns:
            raise ValueError("数据中必须包含 'Returns' 列用于计算奖励。")
        # 保留原始收益，用于奖励计算
        self.raw_returns = df["Returns"].reset_index(drop=True)
        # 使用标准化后的特征作为输入（排除 Returns_standardized）
        std_cols = [col for col in df.columns if col.endswith("_standardized") and col != "Returns_standardized"]
        if not std_cols:
            raise ValueError("未找到标准化特征列")
        self.input_data = df[std_cols].reset_index(drop=True)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.input_data) - self.seq_length

    def __getitem__(self, idx):
        # 取 idx 到 idx+seq_length 的标准化特征作为输入序列
        seq = self.input_data.iloc[idx: idx + self.seq_length].values
        # 取序列后一天的原始收益作为目标
        target_return = self.raw_returns.iloc[idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target_return, dtype=torch.float32)

# ================= 数据集划分函数 =================
def split_dataset(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按时间顺序将数据集划分为训练集、验证集和测试集。
    """
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
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_length, d_model]
        return x + self.pe[:, :x.size(1)]

# ================= Transformer 模型定义 =================
class TransformerModel(nn.Module):
    """
    基于 TransformerEncoder 的模型，用于自监督学习量化交易信号。
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, output_dim: int):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_length, input_dim]
        x = self.embedding(x)  # [batch, seq_length, d_model]
        x = self.positional_encoding(x)
        # TransformerEncoder要求输入形状为 [seq_length, batch, d_model]
        x = x.transpose(0, 1)
        encoded = self.encoder(x)
        # 取最后一个时间步的输出作为序列的表示
        last_output = encoded[-1]  # [batch, d_model]
        signal = self.head(last_output)  # 输出交易信号，形状：[batch, output_dim]
        signal = torch.tanh(signal)  # 限制在 -1~1 之间
        return signal.squeeze(-1)  # 返回形状：[batch]

# ================= 奖励函数 =================
def calculate_reward(target_returns: torch.Tensor, signals: torch.Tensor, risk_penalty: float) -> torch.Tensor:
    """
    计算自监督奖励，基于策略收益（信号与实际收益的乘积）和风险惩罚。
    
    参数:
        target_returns (torch.Tensor): 目标收益（原始 Returns），形状 [batch]
        signals (torch.Tensor): 模型输出的交易信号，形状 [batch]
        risk_penalty (float): 风险惩罚系数。
    
    返回:
        torch.Tensor: 标量奖励
    """
    # 模拟策略收益：信号 * 实际收益
    strategy_returns = signals * target_returns
    # 年化收益（假设每日交易，乘以252个交易日）
    annual_return = strategy_returns.mean() * 252
    # 风险惩罚：策略收益标准差
    risk = torch.std(strategy_returns)
    reward = annual_return - risk_penalty * risk
    return reward

# ================= 训练逻辑 =================
def train_model(model: nn.Module, dataloader: DataLoader, config: dict, model_save_dir: str, device: torch.device, phase="训练", optimizer=None):
    """
    训练或验证/测试模型，使用自监督奖励作为训练目标。
    
    参数:
        model (nn.Module): Transformer 模型。
        dataloader (DataLoader): 数据加载器。
        config (dict): 配置参数。
        model_save_dir (str): 模型保存目录（仅在训练阶段使用）。
        device (torch.device): 训练设备。
        phase (str): 标记阶段，"训练" 或 "验证"/"测试"。
        optimizer (optim.Optimizer): 训练阶段使用的优化器（验证/测试阶段可不传）。
    """
    risk_penalty = config.get("risk_penalty", 0.5)
    
    if phase == "训练":
        if optimizer is None:
            raise ValueError("训练阶段需要传入优化器")
        model.train()
    else:
        model.eval()

    epoch_rewards = []
    for sequences, target_returns in dataloader:
        sequences = sequences.to(device)
        target_returns = target_returns.to(device)

        if phase == "训练":
            optimizer.zero_grad()

        signals = model(sequences)  # [batch]
        reward = calculate_reward(target_returns, signals, risk_penalty)
        
        if phase == "训练":
            loss = -reward  # 最大化奖励
            loss.backward()
            optimizer.step()
        
        epoch_rewards.append(reward.item())

    avg_reward = sum(epoch_rewards) / len(epoch_rewards)
    logger.info(f"{phase}阶段, Average Reward: {avg_reward:.4f}")

    if phase == "训练":
        model_name = f"transformer_model_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth"
        model_save_path = os.path.join(model_save_dir, model_name)
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"模型已保存至: {model_save_path}")

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始训练自监督 Transformer 模型 ==========")

        # 获取项目根目录
        project_root = get_project_root()

        # 数据目录（例如：项目根目录下的 data/standardized）
        data_dir = os.path.join(project_root, "data", "standardized")
        df = load_csv_data(data_dir)
        logger.info(f"数据加载完成，共 {len(df)} 条记录")

        # 小规模试跑机制：如果配置中设置 small_scale 为 True，则只使用部分数据
        config_path = os.path.join(project_root, "configs", "train_selfsupervised_transformer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(f"配置文件加载完成: {config}")

        if config.get("small_scale", False):
            max_samples = config.get("max_samples", 1000)
            df = df.head(max_samples)
            logger.info(f"小规模试跑：使用前 {max_samples} 条记录")

        # 数据集划分：假设数据按时间排序
        train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        logger.info(f"数据集划分完成：训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条, 测试集 {len(test_df)} 条")

        # 创建训练集、验证集、测试集的数据集与 DataLoader
        seq_length = config.get("seq_length", 30)
        batch_size = config.get("batch_size", 64)

        train_dataset = TimeSeriesDataset(train_df, seq_length=seq_length)
        val_dataset = TimeSeriesDataset(val_df, seq_length=seq_length)
        test_dataset = TimeSeriesDataset(test_df, seq_length=seq_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

        # 模型初始化
        input_dim = train_dataset.input_data.shape[1]
        model = TransformerModel(
            input_dim=input_dim,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            output_dim=config["output_dim"]  # 这里建议为 1
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("模型初始化完成")

        # 优化器创建（贯穿整个训练过程）
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # 模型保存目录（例如：项目根目录下的 models）
        model_save_dir = os.path.join(project_root, "models")
        os.makedirs(model_save_dir, exist_ok=True)

        # 训练模型
        for epoch in range(config["num_epochs"]):
            logger.info(f"===== Epoch {epoch+1}/{config['num_epochs']} =====")
            # 训练阶段
            train_model(model, train_loader, config, model_save_dir, device, phase="训练", optimizer=optimizer)
            # 每个 epoch 后进行验证
            train_model(model, val_loader, config, model_save_dir, device, phase="验证")

        # 训练完成后，在测试集上评估模型
        logger.info("========== 模型在测试集上评估 ==========")
        train_model(model, test_loader, config, model_save_dir, device, phase="测试")

        logger.info("========== 训练完成 ==========")

    except Exception as e:
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
