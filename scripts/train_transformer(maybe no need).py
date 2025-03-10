import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
import json

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "transformer_training.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# ================= 数据加载与预处理 =================
def load_and_preprocess_data(input_dir, seq_length=20):
    """
    加载标准化后的数据并构造序列
    :param input_dir: 标准化文件目录
    :param seq_length: 输入序列长度
    :return: X (输入序列), y (交易信号)
    """
    all_X = []
    all_y = []

    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    files = [f for f in os.listdir(input_dir) if f.endswith("_standardized.csv")]
    logger.info(f"找到 {len(files)} 个标准化文件")
    if not files:
        logger.error(f"目录 {input_dir} 中没有找到 _standardized.csv 文件")
        raise ValueError(f"目录 {input_dir} 中没有找到 _standardized.csv 文件")

    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        logger.info(f"处理文件: {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"文件 {file_path} 的前 5 行数据:\n{df.head()}")
            feature_cols = [col for col in df.columns if col != "Date"]
            if not feature_cols:
                logger.error(f"文件 {file_path} 中没有除 Date 外的特征列")
                continue

            # 检查必要列
            required_columns = ["Date", "Open", "Close", "High", "Low", "Volume"]
            if not all(col in df.columns for col in required_columns):
                logger.error(f"文件 {file_path} 缺少必要的列，跳过处理")
                continue

            # 检查 Close 列完整性
            if df["Close"].isnull().any() or (df["Close"] == 0).any():
                logger.error(f"文件 {file_path} 的 Close 列存在缺失值或 0 值，跳过处理")
                continue

            features = df[feature_cols].values

            if len(features) <= seq_length + 1:
                logger.warning(f"文件 {file_path} 数据量不足（{len(features)} 行），无法构造序列（需要至少 {seq_length + 1} 行）")
                continue

            # 定义交易信号（买入：1，卖出：2，持有：0）
            df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
            df["Signal"] = np.where(df["Future_Return"] > 0.005, 1,
                               np.where(df["Future_Return"] < -0.005, 2, 0))
            logger.info(f"文件 {file_path} 的交易信号分布:\n{df['Signal'].value_counts()}")

            X_seq, y_seq = [], []
            # 保证索引不会超出范围
            for i in range(len(features) - seq_length - 1):
                X_seq.append(features[i:i + seq_length])
                y_seq.append(df["Signal"].iloc[i + seq_length])
            
            all_X.extend(X_seq)
            all_y.extend(y_seq)
            logger.info(f"文件 {file_path} 处理完成，生成 {len(X_seq)} 个序列")

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    if not all_X or not all_y:
        logger.error("所有文件处理后未生成任何序列数据，检查文件内容或 seq_length 设置")
        raise ValueError("未生成任何序列数据")

    return np.array(all_X), np.array(all_y)

# ================= Transformer 模型 =================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, num_classes=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # 指定 batch_first=True 以便输入形状为 (batch, seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # 使用序列中最后一个时间步的输出进行分类
        x = self.fc(x[:, -1, :])
        return x

# ================= 训练函数 =================
def train_transformer(X, y, model_path, config):
    """训练 Transformer 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    # 划分训练集和验证集
    val_split = config.get("val_split", 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    input_dim = X.shape[2]
    model = TransformerModel(input_dim, d_model=config["model_params"]["d_model"], 
                             n_heads=config["model_params"]["num_attention_heads"], 
                             n_layers=config["model_params"]["num_layers"], 
                             dropout=config["model_params"]["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        avg_train_loss = train_loss / train_size

        # 验证模型
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(output, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        avg_val_loss = val_loss / val_size
        val_accuracy = correct / total

        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存至: {model_path}")

# ================= 预测与策略生成 =================
def generate_trading_strategy(input_dir, model_path, output_dir, seq_length=20):
    """用训练好的模型生成交易策略"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取第一个文件确认特征数量（不包含 Date 列）
    sample_file = [f for f in os.listdir(input_dir) if f.endswith("_standardized.csv")][0]
    df_sample = pd.read_csv(os.path.join(input_dir, sample_file))
    input_dim = len([col for col in df_sample.columns if col != "Date"])
    
    model = TransformerModel(input_dim, d_model=64, n_heads=4, n_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for file_name in os.listdir(input_dir):
        if file_name.endswith("_standardized.csv"):
            file_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_strategy.csv")
            
            df = pd.read_csv(file_path)
            feature_cols = [col for col in df.columns if col != "Date"]
            features = df[feature_cols].values

            # 生成序列，保证预测的数量与训练时一致（不考虑最后一行标签未知的情况）
            X_seq = [features[i:i + seq_length] for i in range(len(features) - seq_length)]
            if not X_seq:
                logger.warning(f"文件 {file_path} 数据不足，无法生成策略")
                continue
            X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)

            with torch.no_grad():
                preds = model(X_tensor).argmax(dim=1).cpu().numpy()

            # 对应的日期取序列最后一个时刻，从第 seq_length 行开始
            df_strategy = df.iloc[seq_length:].copy()
            df_strategy["Strategy"] = ["Hold" if p == 0 else "Buy" if p == 1 else "Sell" for p in preds]
            df_strategy.to_csv(output_path, index=False)
            logger.info(f"策略已保存至: {output_path}")

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始训练与策略生成 ==========")
        
        input_dir = os.path.join(get_project_root(), "data", "standardized")
        logger.info(f"输入目录: {input_dir}")
        if not os.path.exists(input_dir):
            logger.error(f"输入目录不存在: {input_dir}")
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        model_path = os.path.join(get_project_root(), "models", "transformer.pth")
        output_dir = os.path.join(get_project_root(), "data", "strategy")
        
        # 加载配置文件
        config_path = os.path.join(get_project_root(), "configs", "train_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        X, y = load_and_preprocess_data(input_dir, seq_length=config["seq_len"])
        logger.info(f"数据加载完成，X shape: {X.shape}, y shape: {y.shape}")
        
        train_transformer(X, y, model_path, config)
        
        generate_trading_strategy(input_dir, model_path, output_dir, seq_length=config["seq_len"])
        
        logger.info("========== 训练与策略生成完成 ==========")
    
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()