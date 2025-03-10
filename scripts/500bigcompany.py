import pandas as pd
import json

# 从Wikipedia获取S&P500成分股
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
sp500_df = tables[0]
# 得到股票代码列表
sp500_tickers = sp500_df['Symbol'].tolist()

# 构造配置字典
config = {
    "stocks": sp500_tickers,
    "start_date": "2010-01-01",
    "end_date": "2026-03-01",
    "interval": "1d"
}

# 将配置保存到 JSON 文件中（例如：configs/download_config.json）
with open('configs/download_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print("配置文件已保存。")
