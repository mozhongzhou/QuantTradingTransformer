import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
import json
from concurrent.futures import ThreadPoolExecutor

# ================= 项目路径管理 =================
def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 日志配置 =================
def setup_logger():
    """配置日志记录器"""
    log_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "strategy_generation.log")
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

# ================= 预测与策略生成 =================
def process_file(file_name, input_dir, model, device, seq_length, output_dir):
    """处理单个文件并生成交易策略"""
    file_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_strategy.csv")
    
    df = pd.read_csv(file_path)
    feature_cols = [col for col in df.columns if col != "Date"]
    features = df[feature_cols].values

    # 生成序列，保证预测的数量与训练时一致（不考虑最后一行标签未知的情况）
    X_seq = [features[i:i + seq_length] for i in range(len(features) - seq_length)]
    if not X_seq:
        logger.warning(f"文件 {file_path} 数据不足，无法生成策略")
        return
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).argmax(dim=1).cpu().numpy()

    # 对应的日期取序列最后一个时刻，从第 seq_length 行开始
    df_strategy = df.iloc[seq_length:].copy()
    df_strategy["Strategy"] = ["Hold" if p == 0 else "Buy" if p == 1 else "Sell" for p in preds]
    df_strategy.to_csv(output_path, index=False)
    logger.info(f"策略已保存至: {output_path}")

def generate_trading_strategy(input_dir, model_path, output_dir, seq_length=20):
    """用训练好的模型生成交易策略"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取第一个文件确认特征数量（不包含 Date 列）
    sample_file = [f for f in os.listdir(input_dir) if f.endswith("_standardized.csv")][0]
    df_sample = pd.read_csv(os.path.join(input_dir, sample_file))
    input_dim = len([col for col in df_sample.columns if col != "Date"])
    
    # 加载配置文件
    config_path = os.path.join(get_project_root(), "configs", "train_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model = TransformerModel(input_dim, d_model=config["model_params"]["d_model"], 
                             n_heads=config["model_params"]["num_attention_heads"], 
                             n_layers=config["model_params"]["num_layers"], 
                             dropout=config["model_params"]["dropout"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    files = [f for f in os.listdir(input_dir) if f.endswith("_standardized.csv")]

    # 使用多线程处理文件
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name, input_dir, model, device, seq_length, output_dir) for file_name in files]
        for future in futures:
            future.result()

# ================= 主函数 =================
def main():
    try:
        logger.info("========== 开始生成策略 ==========")
        
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
        
        generate_trading_strategy(input_dir, model_path, output_dir, seq_length=config["seq_len"])
        
        logger.info("========== 策略生成完成 ==========")
    
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()