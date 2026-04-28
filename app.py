
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Pro Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0b0f19; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #2d3748;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.25);
    }
    div[data-testid="stMetricLabel"] { color: #9ca3af; }
    div[data-testid="stMetricValue"] { color: #f9fafb; }
    .signal-buy {
        background: linear-gradient(135deg, #064e3b 0%, #10b981 100%);
        padding: 22px; border-radius: 18px; color: white;
        font-size: 28px; font-weight: 800; text-align: center;
        box-shadow: 0 12px 30px rgba(16,185,129,.22);
    }
    .signal-sell {
        background: linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%);
        padding: 22px; border-radius: 18px; color: white;
        font-size: 28px; font-weight: 800; text-align: center;
        box-shadow: 0 12px 30px rgba(239,68,68,.22);
    }
    .signal-hold {
        background: linear-gradient(135deg, #78350f 0%, #f59e0b 100%);
        padding: 22px; border-radius: 18px; color: white;
        font-size: 28px; font-weight: 800; text-align: center;
        box-shadow: 0 12px 30px rgba(245,158,11,.22);
    }
    .card {
        background: #111827;
        border: 1px solid #2d3748;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 14px;
    }
    .small-muted { color: #9ca3af; font-size: 13px; }
    .definition { color: #cbd5e1; font-size: 13px; margin-top: 7px; line-height: 1.35; }
    .big-number { color: #f9fafb; font-size: 32px; font-weight: 800; }
    .section-title {
        color: #f9fafb; font-size: 22px; font-weight: 800;
        margin-top: 14px; margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFS = {
    "Last Price": "Most recent closing/market price returned by Yahoo Finance.",
    "Signal": "The app’s combined Buy, Sell/Avoid, or Hold/Wait reading based on technical indicators.",
    "Confidence": "How strongly the indicators agree. Higher confidence means stronger technical alignment, not a guarantee.",
    "Score": "Internal point system combining trend, momentum, volume, RSI, and volatility.",
    "RSI 14": "Relative Strength Index over 14 periods. Above 70 can be overbought; below 30 can be oversold.",
    "ATR 14": "Average True Range over 14 periods. Measures normal price movement/volatility.",
    "Volatility": "ATR divided by price. Higher percentage means the stock is moving more aggressively.",
    "Support": "Recent lower price zone where buyers may step in.",
    "Resistance": "Recent upper price zone where sellers may step in.",
    "MACD": "Moving Average Convergence Divergence. A trend/momentum indicator.",
    "Signal Line": "MACD’s smoothing line. MACD above this line is usually bullish; below is usually bearish.",
    "Suggested Order Type": "The app’s preferred order style based on signal and volatility.",
    "Preferred Entry Zone": "Estimated buy zone based on current price and ATR.",
    "Stop Loss": "A planned sell level used to limit losses if the trade moves against you.",
    "Take Profit": "A planned sell level used to lock in gains if price reaches the target.",
    "Limit Buy": "An order to buy only at your chosen price or better.",
    "Market Buy": "An order that buys immediately at the best available current price.",
    "Stop-Limit Breakout": "An order that triggers near a breakout price, then uses a limit price to avoid overpaying.",
    "Stop-Limit Sell": "A protective sell order that triggers at a stop price, then sells at your limit or better.",
    "No Trade / Alert": "No order is suggested. Set alerts and wait for cleaner confirmation."
}

def def_text(name):
    return DEFS.get(name, "")

def metric_with_definition(label, value, definition, delta=None):
    st.metric(label, value, delta=delta, help=definition)
    st.markdown(f"<div class='definition'>{definition}</div>", unsafe_allow_html=True)

def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def bollinger(series, period=20, std_mult=2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"])
    macd_line, macd_signal, macd_hist = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACDSignal"] = macd_signal
    df["MACDHist"] = macd_hist
    df["ATR14"] = atr(df)
    df["BBUpper"], df["BBMid"], df["BBLower"] = bollinger(df["Close"])
    df["VolumeAvg20"] = df["Volume"].rolling(20).mean()
    return df

def score_setup(df, mode, risk):
    latest = df.dropna().iloc[-1]
    prev = df.dropna().iloc[-2]
    price = float(latest["Close"])
    atr_val = float(latest["ATR14"])
    volatility_pct = atr_val / price if price else 0

    score = 0
    max_score = 11
    bullish, bearish, neutral = [], [], []

    if price > latest["SMA20"]:
        score += 1; bullish.append("Price is above the 20-day moving average.")
    else:
        score -= 1; bearish.append("Price is below the 20-day moving average.")

    if latest["SMA20"] > latest["SMA50"]:
        score += 1; bullish.append("20-day average is above the 50-day average.")
    else:
        score -= 1; bearish.append("20-day average is below the 50-day average.")

    if price > latest["SMA200"]:
        score += 1; bullish.append("Price is above the 200-day moving average.")
    else:
        score -= 1; bearish.append("Price is below the 200-day moving average.")

    if latest["EMA9"] > latest["EMA21"]:
        score += 1; bullish.append("9 EMA is above the 21 EMA.")
    else:
        score -= 1; bearish.append("9 EMA is below the 21 EMA.")

    if latest["MACD"] > latest["MACDSignal"]:
        score += 1; bullish.append("MACD is above its signal line.")
    else:
        score -= 1; bearish.append("MACD is below its signal line.")

    if latest["MACDHist"] > prev["MACDHist"]:
        score += 1; bullish.append("MACD momentum is improving.")
    else:
        score -= 1; bearish.append("MACD momentum is weakening.")

    if 45 <= latest["RSI14"] <= 65:
        score += 1; bullish.append("RSI is in a healthy momentum zone.")
    elif latest["RSI14"] < 30:
        score += 1; bullish.append("RSI is oversold, possible bounce setup.")
    elif latest["RSI14"] > 75:
        score -= 1; bearish.append("RSI is very overbought.")
    else:
        neutral.append("RSI is not giving a strong signal.")

    if latest["Volume"] > latest["VolumeAvg20"] * 1.15:
        score += 1; bullish.append("Volume is above average, confirming interest.")
    elif latest["Volume"] < latest["VolumeAvg20"] * 0.75:
        score -= 1; bearish.append("Volume is weak compared with average.")
    else:
        neutral.append("Volume is near normal.")

    if price <= latest["BBLower"]:
        score += 1; bullish.append("Price is near/below the lower Bollinger Band.")
    elif price >= latest["BBUpper"]:
        score -= 1; bearish.append("Price is near/above the upper Bollinger Band.")
    else:
        neutral.append("Price is inside the Bollinger Bands.")

    if mode == "Day Trade":
        if volatility_pct > 0.025:
            score += 1; bullish.append("Volatility is high enough for active trading.")
        else:
            neutral.append("Volatility may be low for day trading.")
    elif mode == "Swing Trade":
        if latest["SMA20"] > latest["SMA50"] and latest["MACD"] > latest["MACDSignal"]:
            score += 1; bullish.append("Swing trend and momentum are aligned.")
    else:
        if price > latest["SMA200"] and latest["SMA50"] > latest["SMA200"]:
            score += 1; bullish.append("Long-term trend structure is constructive.")

    if score >= 5:
        signal, signal_class = "BUY", "signal-buy"
    elif score <= -3:
        signal, signal_class = "SELL / AVOID", "signal-sell"
    else:
        signal, signal_class = "HOLD / WAIT", "signal-hold"

    confidence = min(95, max(5, int((abs(score) / max_score) * 100 + 35)))

    stop_mult, target_mult = {
        "Conservative": (1.5, 2.0),
        "Balanced": (2.0, 3.0),
        "Aggressive": (2.5, 4.0)
    }[risk]

    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    if signal == "BUY":
        entry_low = price - 0.35 * atr_val
        entry_high = price + 0.10 * atr_val
        if volatility_pct > 0.04:
            order_type = "LIMIT BUY"
            order_reason = "Volatility is elevated, so a limit order helps avoid chasing a spike."
        elif mode == "Day Trade":
            order_type = "LIMIT BUY or STOP-LIMIT BREAKOUT"
            order_reason = "Use a limit near pullback support, or stop-limit above resistance if trading a breakout."
        else:
            order_type = "LIMIT BUY"
            order_reason = "The setup is bullish, but a limit order gives better entry control."
        stop_loss = max(price - stop_mult * atr_val, support - 0.25 * atr_val)
        take_profit = price + target_mult * atr_val
    elif signal == "SELL / AVOID":
        entry_low = np.nan; entry_high = np.nan
        order_type = "STOP LOSS / STOP-LIMIT SELL"
        order_reason = "The setup is weak. If already holding, use a protective stop. If not holding, avoid entry."
        stop_loss = price - 1.25 * atr_val
        take_profit = np.nan
    else:
        entry_low = price - 0.5 * atr_val
        entry_high = price + 0.5 * atr_val
        order_type = "NO TRADE / PRICE ALERT"
        order_reason = "Signals are mixed. Wait for confirmation before entering."
        stop_loss = np.nan
        take_profit = np.nan

    return {
        "signal": signal, "signal_class": signal_class, "confidence": confidence,
        "score": score, "price": price, "atr": atr_val, "volatility_pct": volatility_pct,
        "rsi": float(latest["RSI14"]), "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACDSignal"]), "support": support,
        "resistance": resistance, "entry_low": entry_low, "entry_high": entry_high,
        "order_type": order_type, "order_reason": order_reason,
        "stop_loss": stop_loss, "take_profit": take_profit,
        "bullish": bullish, "bearish": bearish, "neutral": neutral
    }

def make_chart(df, ticker):
    plot_df = df.dropna().tail(180)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.62, 0.18, 0.20]
    )
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA 50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BBUpper"], name="BB Upper", line=dict(width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BBLower"], name="BB Lower", line=dict(width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI14"], name="RSI 14", line=dict(width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", row=3, col=1)
    fig.update_layout(
        title=f"{ticker} Price Action",
        template="plotly_dark", height=780,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0b0f19", plot_bgcolor="#111827",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

with st.sidebar:
    st.header("Analyzer Settings")
    ticker = st.text_input("Ticker", value="AAPL", help="Stock symbol, such as AAPL, NVDA, TSLA, SPY, or AMD.").upper().strip()
    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2, help="How much past price history to analyze.")
    mode = st.selectbox("Trading mode", ["Day Trade", "Swing Trade", "Long-Term"], index=1, help="Changes how the signal is weighted.")
    risk = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1, help="Changes stop-loss and take-profit distance.")
    st.divider()
    st.caption("Definitions are shown under the metrics and in hover tooltips.")

st.markdown("# 📈 Pro Stock Analyzer")
st.markdown("Buy/Sell/Hold scoring, suggested order type, technical indicators, trade plan zones, and definitions beside each value.")
st.markdown("<div class='small-muted'>Educational tool only. Not financial advice.</div>", unsafe_allow_html=True)

if not ticker:
    st.warning("Enter a ticker to start.")
    st.stop()

try:
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    data = clean_columns(data)

    if data.empty:
        st.error("No price data found. Check the ticker symbol.")
        st.stop()

    df = add_indicators(data)
    if len(df.dropna()) < 30:
        st.error("Not enough historical data to analyze this ticker.")
        st.stop()

    result = score_setup(df, mode, risk)

    col_signal, col_metrics = st.columns([1.2, 2.8])

    with col_signal:
        st.markdown(f"<div class='{result['signal_class']}'>{result['signal']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='definition'><b>Signal:</b> {DEFS['Signal']}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Confidence</div>
            <div class='big-number'>{result['confidence']}%</div>
            <div class='definition'><b>Confidence:</b> {DEFS['Confidence']}</div>
            <div class='small-muted' style='margin-top:8px;'>Score: {result['score']}</div>
            <div class='definition'><b>Score:</b> {DEFS['Score']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_with_definition("Last Price", f"${result['price']:.2f}", DEFS["Last Price"])
        with m2: metric_with_definition("RSI 14", f"{result['rsi']:.1f}", DEFS["RSI 14"])
        with m3: metric_with_definition("ATR 14", f"${result['atr']:.2f}", DEFS["ATR 14"])
        with m4: metric_with_definition("Volatility", f"{result['volatility_pct']*100:.2f}%", DEFS["Volatility"])

        m5, m6, m7, m8 = st.columns(4)
        with m5: metric_with_definition("Support", f"${result['support']:.2f}", DEFS["Support"])
        with m6: metric_with_definition("Resistance", f"${result['resistance']:.2f}", DEFS["Resistance"])
        with m7: metric_with_definition("MACD", f"{result['macd']:.2f}", DEFS["MACD"])
        with m8: metric_with_definition("Signal Line", f"{result['macd_signal']:.2f}", DEFS["Signal Line"])

    st.markdown("<div class='section-title'>Suggested Trade Plan</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Suggested Order Type</div>
            <div class='big-number' style='font-size:24px;'>{result['order_type']}</div>
            <div class='definition'><b>Suggested Order Type:</b> {DEFS['Suggested Order Type']}</div>
            <div class='definition'><b>Why:</b> {result['order_reason']}</div>
            <div class='definition'><b>Order definition:</b> {DEFS.get('Limit Buy') if 'LIMIT BUY' in result['order_type'] else DEFS.get('Stop-Limit Sell') if 'STOP' in result['order_type'] else DEFS.get('No Trade / Alert')}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        entry = f"${result['entry_low']:.2f} - ${result['entry_high']:.2f}" if not np.isnan(result["entry_low"]) else "No new entry"
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Preferred Entry Zone</div>
            <div class='big-number' style='font-size:24px;'>{entry}</div>
            <div class='definition'><b>Preferred Entry Zone:</b> {DEFS['Preferred Entry Zone']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        stop = f"${result['stop_loss']:.2f}" if not np.isnan(result["stop_loss"]) else "N/A"
        target = f"${result['take_profit']:.2f}" if not np.isnan(result["take_profit"]) else "N/A"
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Risk / Reward</div>
            <div style='color:#f9fafb; font-size:20px; font-weight:700;'>Stop Loss: {stop}</div>
            <div class='definition'><b>Stop Loss:</b> {DEFS['Stop Loss']}</div>
            <div style='color:#f9fafb; font-size:20px; font-weight:700; margin-top:8px;'>Take Profit: {target}</div>
            <div class='definition'><b>Take Profit:</b> {DEFS['Take Profit']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.plotly_chart(make_chart(df, ticker), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Bullish Factors")
        for item in result["bullish"]:
            st.success(item)
        if not result["bullish"]:
            st.info("No strong bullish factors detected.")

    with right:
        st.markdown("### Bearish / Caution Factors")
        for item in result["bearish"]:
            st.error(item)
        if not result["bearish"]:
            st.info("No major bearish factors detected.")

    with st.expander("Definitions Cheat Sheet"):
        for k, v in DEFS.items():
            st.write(f"**{k}:** {v}")

    with st.expander("Recent Indicator Data"):
        display_cols = [
            "Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "SMA200",
            "EMA9", "EMA21", "RSI14", "MACD", "MACDSignal", "MACDHist", "ATR14"
        ]
        st.dataframe(df[display_cols].tail(30), use_container_width=True)

    st.divider()
    st.caption("Important: This app uses technical indicators only and does not consider your portfolio, taxes, account size, earnings, news, or macro conditions.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
