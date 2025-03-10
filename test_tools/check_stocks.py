import yfinance as yf

stocks = ["CEG", "CRWD", "CVS", "DGX", "DXCM", "GDDY", "HBAN", "HUBB", "HWM"]
for stock in stocks:
    ticker = yf.Ticker(stock)
    history = ticker.history(period="max")  # 获取最长历史数据
    if not history.empty:
        first_date = history.index[0].strftime("%Y-%m-%d")
        print(f"{stock} 的最早数据日期: {first_date}")
    else:
        print(f"{stock} 无可用数据")