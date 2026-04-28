
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="Pro Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main {
        background-color: #0b0f19;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #2d3748;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
    div[data-testid="stMetricValue"] {
        color: #f9fafb;
    }
    .signal-buy {
        background: linear-gradient(135deg, #064e3b 0%, #10b981 100%);
        padding: 22px;
        border-radius: 18px;
        color: white;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 12px 30px rgba(16,185,129,.22);
    }
    .signal-sell {
        background: linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%);
        padding: 22px;
        border-radius: 18px;
        color: white;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 12px 30px rgba(239,68,68,.22);
    }
    .signal-hold {
        background: linear-gradient(135deg, #78350f 0%, #f59e0b 100%);
        padding: 22px;
        border-radius: 18px;
        color: white;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 12px 30px rgba(245,158,11,.22);
    }
    .card {
        background: #111827;
        border: 1px solid #2d3748;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 14px;
    }
    .small-muted {
        color: #9ca3af;
        font-size: 13px;
    }
    .big-number {
        color: #f9fafb;
        font-size: 32px;
        font-weight: 800;
    }
    .section-title {
        color: #f9fafb;
        font-size: 22px;
        font-weight: 800;
        margin-top: 14px;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    df["DailyReturn"] = df["Close"].pct_change()
    return df

def score_setup(df: pd.DataFrame, mode: str, risk: str):
    latest = df.dropna().iloc[-1]
    prev = df.dropna().iloc[-2]
    price = float(latest["Close"])
    atr_val = float(latest["ATR14"])
    volatility_pct = atr_val / price if price else 0

    score = 0
    max_score = 11
    bullish = []
    bearish = []
    neutral = []

    # Trend
    if price > latest["SMA20"]:
        score += 1
        bullish.append("Price is above the 20-day moving average.")
    else:
        score -= 1
        bearish.append("Price is below the 20-day moving average.")

    if latest["SMA20"] > latest["SMA50"]:
        score += 1
        bullish.append("20-day average is above the 50-day average.")
    else:
        score -= 1
        bearish.append("20-day average is below the 50-day average.")

    if price > latest["SMA200"]:
        score += 1
        bullish.append("Price is above the 200-day moving average.")
    else:
        score -= 1
        bearish.append("Price is below the 200-day moving average.")

    # Momentum
    if latest["EMA9"] > latest["EMA21"]:
        score += 1
        bullish.append("9 EMA is above the 21 EMA.")
    else:
        score -= 1
        bearish.append("9 EMA is below the 21 EMA.")

    if latest["MACD"] > latest["MACDSignal"]:
        score += 1
        bullish.append("MACD is above signal line.")
    else:
        score -= 1
        bearish.append("MACD is below signal line.")

    if latest["MACDHist"] > prev["MACDHist"]:
        score += 1
        bullish.append("MACD momentum is improving.")
    else:
        score -= 1
        bearish.append("MACD momentum is weakening.")

    # RSI logic
    if 45 <= latest["RSI14"] <= 65:
        score += 1
        bullish.append("RSI is in a healthy momentum zone.")
    elif latest["RSI14"] < 30:
        score += 1
        bullish.append("RSI is oversold, possible bounce setup.")
    elif latest["RSI14"] > 75:
        score -= 1
        bearish.append("RSI is very overbought.")
    else:
        neutral.append("RSI is not giving a strong signal.")

    # Volume
    if latest["Volume"] > latest["VolumeAvg20"] * 1.15:
        score += 1
        bullish.append("Volume is above average, confirming interest.")
    elif latest["Volume"] < latest["VolumeAvg20"] * 0.75:
        score -= 1
        bearish.append("Volume is weak compared with average.")
    else:
        neutral.append("Volume is near normal.")

    # Bollinger location
    if price <= latest["BBLower"]:
        score += 1
        bullish.append("Price is near/below the lower Bollinger Band.")
    elif price >= latest["BBUpper"]:
        score -= 1
        bearish.append("Price is near/above the upper Bollinger Band.")
    else:
        neutral.append("Price is inside the Bollinger Bands.")

    # Mode adjustment
    if mode == "Day Trade":
        if volatility_pct > 0.025:
            score += 1
            bullish.append("Volatility is high enough for active trading.")
        else:
            neutral.append("Volatility may be low for day trading.")
    elif mode == "Swing Trade":
        if latest["SMA20"] > latest["SMA50"] and latest["MACD"] > latest["MACDSignal"]:
            score += 1
            bullish.append("Swing trend and momentum are aligned.")
    else:  # Long-Term
        if price > latest["SMA200"] and latest["SMA50"] > latest["SMA200"]:
            score += 1
            bullish.append("Long-term trend structure is constructive.")

    if score >= 5:
        signal = "BUY"
        signal_class = "signal-buy"
    elif score <= -3:
        signal = "SELL / AVOID"
        signal_class = "signal-sell"
    else:
        signal = "HOLD / WAIT"
        signal_class = "signal-hold"

    confidence = min(95, max(5, int((abs(score) / max_score) * 100 + 35)))

    if risk == "Conservative":
        stop_mult = 1.5
        target_mult = 2.0
    elif risk == "Aggressive":
        stop_mult = 2.5
        target_mult = 4.0
    else:
        stop_mult = 2.0
        target_mult = 3.0

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
        invalidation = stop_loss

    elif signal == "SELL / AVOID":
        entry_low = np.nan
        entry_high = np.nan
        order_type = "STOP LOSS / STOP-LIMIT SELL"
        order_reason = "The setup is weak. If already holding, use a protective stop. If not holding, avoid entry."
        stop_loss = price - 1.25 * atr_val
        take_profit = np.nan
        invalidation = price + 1.0 * atr_val

    else:
        entry_low = price - 0.5 * atr_val
        entry_high = price + 0.5 * atr_val
        order_type = "NO TRADE / PRICE ALERT"
        order_reason = "Signals are mixed. Wait for confirmation before entering."
        stop_loss = np.nan
        take_profit = np.nan
        invalidation = np.nan

    return {
        "signal": signal,
        "signal_class": signal_class,
        "confidence": confidence,
        "score": score,
        "price": price,
        "atr": atr_val,
        "volatility_pct": volatility_pct,
        "rsi": float(latest["RSI14"]),
        "macd": float(latest["MACD"]),
        "macd_signal": float(latest["MACDSignal"]),
        "support": support,
        "resistance": resistance,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "order_type": order_type,
        "order_reason": order_reason,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "invalidation": invalidation,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral
    }

def make_chart(df: pd.DataFrame, ticker: str):
    plot_df = df.dropna().tail(180)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.18, 0.20],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="Price"
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
        template="plotly_dark",
        height=780,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

with st.sidebar:
    st.header("Analyzer Settings")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2)
    mode = st.selectbox("Trading mode", ["Day Trade", "Swing Trade", "Long-Term"], index=1)
    risk = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    st.divider()
    st.caption("Tip: Use liquid tickers with enough trading history. Examples: AAPL, NVDA, TSLA, SPY, AMD.")

st.markdown("# 📈 Pro Stock Analyzer")
st.markdown("Buy/Sell/Hold scoring, suggested order type, technical indicators, and trade plan zones.")
st.markdown("<div class='small-muted'>Educational tool only. Not financial advice. Always verify with your own research and risk controls.</div>", unsafe_allow_html=True)

if not ticker:
    st.warning("Enter a ticker to start.")
    st.stop()

try:
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    data = clean_columns(data)

    if data.empty:
        st.error("No price data found. Check the ticker symbol.")
        st.stop()

    if len(data) < 220 and period in ["1y", "2y", "5y"]:
        st.warning("Limited data returned. Some long-term indicators may be less reliable.")

    df = add_indicators(data)
    usable = df.dropna()

    if len(usable) < 30:
        st.error("Not enough historical data to analyze this ticker.")
        st.stop()

    result = score_setup(df, mode, risk)

    col_signal, col_metrics = st.columns([1.2, 2.8])
    with col_signal:
        st.markdown(f"<div class='{result['signal_class']}'>{result['signal']}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Confidence</div>
            <div class='big-number'>{result['confidence']}%</div>
            <div class='small-muted'>Score: {result['score']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last Price", f"${result['price']:.2f}")
        m2.metric("RSI 14", f"{result['rsi']:.1f}")
        m3.metric("ATR 14", f"${result['atr']:.2f}")
        m4.metric("Volatility", f"{result['volatility_pct']*100:.2f}%")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Support", f"${result['support']:.2f}")
        m6.metric("Resistance", f"${result['resistance']:.2f}")
        m7.metric("MACD", f"{result['macd']:.2f}")
        m8.metric("Signal Line", f"{result['macd_signal']:.2f}")

    st.markdown("<div class='section-title'>Suggested Trade Plan</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Suggested Order Type</div>
            <div class='big-number' style='font-size:24px;'>{result['order_type']}</div>
            <div class='small-muted'>{result['order_reason']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if not np.isnan(result["entry_low"]):
            entry = f"${result['entry_low']:.2f} - ${result['entry_high']:.2f}"
        else:
            entry = "No new entry"
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Preferred Entry Zone</div>
            <div class='big-number' style='font-size:24px;'>{entry}</div>
            <div class='small-muted'>Based on current price and ATR.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        stop = f"${result['stop_loss']:.2f}" if not np.isnan(result["stop_loss"]) else "N/A"
        target = f"${result['take_profit']:.2f}" if not np.isnan(result["take_profit"]) else "N/A"
        st.markdown(f"""
        <div class='card'>
            <div class='small-muted'>Risk / Reward</div>
            <div style='color:#f9fafb; font-size:20px; font-weight:700;'>Stop: {stop}</div>
            <div style='color:#f9fafb; font-size:20px; font-weight:700;'>Target: {target}</div>
            <div class='small-muted'>Adjust position size before placing any trade.</div>
        </div>
        """, unsafe_allow_html=True)

    st.plotly_chart(make_chart(df, ticker), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Bullish Factors")
        if result["bullish"]:
            for item in result["bullish"]:
                st.success(item)
        else:
            st.info("No strong bullish factors detected.")

    with right:
        st.markdown("### Bearish / Caution Factors")
        if result["bearish"]:
            for item in result["bearish"]:
                st.error(item)
        else:
            st.info("No major bearish factors detected.")

    with st.expander("Neutral / Mixed Notes"):
        for item in result["neutral"]:
            st.write("- " + item)

    with st.expander("Recent Indicator Data"):
        display_cols = [
            "Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "SMA200",
            "EMA9", "EMA21", "RSI14", "MACD", "MACDSignal", "MACDHist", "ATR14"
        ]
        st.dataframe(df[display_cols].tail(30), use_container_width=True)

    st.divider()
    st.caption(
        "Important: This app uses technical indicators only. It does not know your portfolio, tax situation, "
        "risk tolerance, account size, news events, earnings risk, or macro conditions."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
