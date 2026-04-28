# Pro Stock Analyzer Browser App

This is a browser-based Streamlit stock analysis app.

## Features

- Buy / Sell / Hold signal
- Confidence score
- Suggested order type
- Entry zone
- Stop loss
- Take profit
- Support and resistance
- Candlestick chart
- SMA 20 / SMA 50 / SMA 200
- EMA 9 / EMA 21
- RSI
- MACD
- Bollinger Bands
- ATR volatility
- Volume confirmation
- Day Trade / Swing Trade / Long-Term modes

## Run locally

Install Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deploy to Streamlit Cloud

1. Create a free GitHub account.
2. Create a new GitHub repository.
3. Upload these files:
   - app.py
   - requirements.txt
   - README.md
4. Go to Streamlit Cloud.
5. Click New App.
6. Connect your GitHub repo.
7. Set main file path to:

```text
app.py
```

8. Deploy.

You will get a public browser link you can open from phone or computer.

## Disclaimer

Educational tool only. This is not financial advice.
