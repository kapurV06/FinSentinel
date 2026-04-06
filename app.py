"""
FinSentinel — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings, json
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Bloomberg terminal meets modern dark UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0a0c10 !important;
    color: #e2e8f0 !important;
}

/* App background */
.stApp {
    background: #0a0c10;
}

/* Main header */
.fin-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.fin-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent, #38bdf8);
}
.metric-label {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-delta {
    font-size: 0.7rem;
    margin-top: 0.3rem;
}

/* Risk badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-high     { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.badge-medium   { background: rgba(234,179,8,0.15);  color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.badge-low      { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #64748b !important; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    color: #38bdf8;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e293b;
}

/* Alert box */
.alert-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
}
.alert-box-safe {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
}

/* Plotly chart container */
.js-plotly-plot { border-radius: 8px; }

/* DataFrame */
.stDataFrame { border: 1px solid #1e293b; border-radius: 8px; }

/* Button */
.stButton > button {
    background: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #334155;
    border-color: #38bdf8;
    color: #f1f5f9;
}

/* Chat input */
.stChatInput input {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Spinner */
.stSpinner { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "TSLA", "AMZN", "MSFT", "GOOGL"]
START_DATE     = "2020-01-01"
END_DATE       = datetime.today().strftime("%Y-%m-%d")
SEQUENCE_LEN   = 20
CONTAMINATION  = 0.05
ALERT_THRESH   = 0.95

COLORS = {
    "primary":  "#38bdf8",
    "accent":   "#818cf8",
    "danger":   "#ef4444",
    "warning":  "#f97316",
    "success":  "#22c55e",
    "bg":       "#0a0c10",
    "card":     "#111827",
    "border":   "#1e293b",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        gridcolor="#1e293b",
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        margin=dict(t=40, b=40, l=50, r=20),
    )
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline functions (cached for performance)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def compute_features(df):
    f = pd.DataFrame(index=df.index)
    close = df["Close"]; vol = df["Volume"]

    f["log_returns"]   = np.log(close / close.shift(1))
    f["volatility_5"]  = f["log_returns"].rolling(5).std()
    f["volatility_20"] = f["log_returns"].rolling(20).std()
    f["vol_ratio"]     = f["volatility_5"] / (f["volatility_20"] + 1e-9)

    ma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
    ma50 = close.rolling(50).mean()
    f["z_vs_ma20"] = (close - ma20) / (std20 + 1e-9)
    f["z_vs_ma50"] = (close - ma50) / (close.rolling(50).std() + 1e-9)
    f["hl_range"]  = (df["High"] - df["Low"]) / (close + 1e-9)
    f["hl_ratio"]  = f["hl_range"] / (f["hl_range"].rolling(10).mean() + 1e-9)
    f["gap"]       = (df["Open"] - close.shift(1)) / (close.shift(1) + 1e-9)

    vol_ma = vol.rolling(20).mean()
    f["volume_ratio"] = vol / (vol_ma + 1e-9)
    f["volume_z"]     = (vol - vol_ma) / (vol.rolling(20).std() + 1e-9)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))
    f["rsi_norm"]  = (rsi - 50) / 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    f["macd_norm"] = (ema12 - ema26) / (close + 1e-9)
    f["bb_pos"]    = (close - ma20) / (2 * std20 + 1e-9)
    f["above_ma50"]= (close > ma50).astype(float)

    return f.dropna()


@st.cache_resource(show_spinner=False)
def run_pipeline(ticker):
    """Full ML+DL pipeline — cached per ticker."""
    df   = fetch_data(ticker, START_DATE, END_DATE)
    feat = compute_features(df)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(feat)

    # ── Isolation Forest ──────────────────────────────────────────────────────
    clf = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                          random_state=42, n_jobs=-1)
    clf.fit(X_sc)
    raw_if  = clf.score_samples(X_sc)
    if_sc   = 1 - (raw_if - raw_if.min()) / (raw_if.ptp() + 1e-9)

    # ── LSTM Autoencoder ──────────────────────────────────────────────────────
    seqs   = np.array([X_sc[i:i+SEQUENCE_LEN] for i in range(len(X_sc)-SEQUENCE_LEN)])
    n_feat = X_sc.shape[1]

    inp  = keras.Input(shape=(SEQUENCE_LEN, n_feat))
    x    = layers.LSTM(64, return_sequences=True)(inp)
    x    = layers.Dropout(0.15)(x)
    x    = layers.LSTM(32)(x)
    x    = layers.RepeatVector(SEQUENCE_LEN)(x)
    x    = layers.LSTM(32, return_sequences=True)(x)
    x    = layers.LSTM(64, return_sequences=True)(x)
    out  = layers.TimeDistributed(layers.Dense(n_feat))(x)
    ae   = keras.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    split = int(0.8 * len(seqs))
    ae.fit(seqs[:split], seqs[:split],
           epochs=30, batch_size=32,
           validation_split=0.1, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    recon  = ae.predict(seqs, verbose=0)
    mse    = np.mean((seqs - recon) ** 2, axis=(1, 2))
    lstm_sc = (mse - mse.min()) / (mse.ptp() + 1e-9)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    min_len = min(len(if_sc), len(lstm_sc))
    ens     = 0.4 * if_sc[-min_len:] + 0.6 * lstm_sc[-min_len:]
    dates   = feat.index[SEQUENCE_LEN:][-min_len:]

    scores  = pd.Series(ens, index=dates, name="anomaly_score")
    thr     = scores.quantile(ALERT_THRESH)

    return {
        "df":      df,
        "feat":    feat,
        "scores":  scores,
        "if_sc":   pd.Series(if_sc[-min_len:], index=dates),
        "lstm_sc": pd.Series(lstm_sc, index=dates),
        "thr":     thr,
        "anomalies": scores[scores > thr],
    }


def risk_level(score, thr):
    if   score > thr * 1.25: return "CRITICAL", "badge-critical", COLORS["danger"]
    elif score > thr:         return "HIGH",     "badge-high",     COLORS["warning"]
    elif score > thr * 0.80:  return "MEDIUM",   "badge-medium",   "#eab308"
    else:                     return "LOW",       "badge-low",      COLORS["success"]


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────
def price_chart(data, ticker):
    df     = data["df"]
    scores = data["scores"]
    anom   = data["anomalies"]
    thr    = data["thr"]

    idx    = scores.index.intersection(df.index)
    price  = df["Close"][idx]
    anom_p = price[price.index.isin(anom.index)]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.27, 0.18],
                        vertical_spacing=0.04)

    # Candlestick (sampled to keep it fast)
    d = df.last("500D")
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"],
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
        name="OHLC", showlegend=False
    ), row=1, col=1)

    # Anomaly markers
    ap = anom_p[anom_p.index >= d.index[0]]
    fig.add_trace(go.Scatter(
        x=ap.index, y=ap * 1.01,
        mode="markers", name="⚠ Anomaly",
        marker=dict(color=COLORS["danger"], size=10, symbol="triangle-up",
                    line=dict(color="white", width=0.5)),
    ), row=1, col=1)

    # Ensemble score
    sc = scores[scores.index >= d.index[0]]
    fig.add_trace(go.Scatter(
        x=sc.index, y=sc,
        name="Anomaly Score",
        line=dict(color=COLORS["accent"], width=1.5),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.08)"
    ), row=2, col=1)
    fig.add_hline(y=thr, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=f"95th pct = {thr:.3f}",
                  annotation_font_color=COLORS["danger"],
                  annotation_position="top right", row=2, col=1)

    # Volume
    vr = data["feat"]["volume_ratio"]
    vr = vr[vr.index >= d.index[0]]
    colors_vol = [COLORS["danger"] if v > 2 else COLORS["primary"] for v in vr]
    fig.add_trace(go.Bar(
        x=vr.index, y=vr, name="Vol Ratio",
        marker_color=colors_vol, opacity=0.8
    ), row=3, col=1)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=680,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, font_color="#94a3b8"),
        title=dict(text=f"<b>{ticker}</b> — Anomaly Detection Dashboard",
                   font=dict(color="#f1f5f9", size=14, family="Syne")),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     title_font_color="#475569", tickfont_color="#64748b")
    fig.update_yaxes(title_text="Score", row=2, col=1,
                     title_font_color="#475569", tickfont_color="#64748b")
    fig.update_yaxes(title_text="Vol/MA", row=3, col=1,
                     title_font_color="#475569", tickfont_color="#64748b")
    return fig


def portfolio_chart(results):
    tickers = list(results.keys())
    scores  = [float(results[t]["scores"].iloc[-1]) for t in tickers]
    thrs    = [float(results[t]["thr"]) for t in tickers]
    normed  = [s / thr for s, thr in zip(scores, thrs)]
    colors  = [COLORS["danger"] if n > 1.25
               else COLORS["warning"] if n > 1.0
               else "#eab308" if n > 0.8
               else COLORS["success"] for n in normed]

    fig = go.Figure(go.Bar(
        x=tickers, y=normed,
        marker_color=colors, opacity=0.85,
        text=[f"{v:.2f}x" for v in normed],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Alert threshold",
                  annotation_font_color=COLORS["danger"],
                  annotation_position="top left")
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=320,
        title=dict(text="Portfolio Risk Comparison (score / threshold)",
                   font=dict(color="#f1f5f9", size=13, family="Syne")),
        yaxis_title="Normalised Risk",
        showlegend=False,
    )
    return fig


def score_history_chart(data, ticker):
    sc  = data["scores"].last("90D")
    isc = data["if_sc"].last("90D")
    lst = data["lstm_sc"].last("90D")
    thr = data["thr"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc.index,  y=sc,  name="Ensemble",
                             line=dict(color=COLORS["accent"], width=2)))
    fig.add_trace(go.Scatter(x=isc.index, y=isc, name="Isolation Forest",
                             line=dict(color=COLORS["primary"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=lst.index, y=lst, name="LSTM AE",
                             line=dict(color="#fb7185", width=1, dash="dot")))
    fig.add_hline(y=thr, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=f"threshold={thr:.3f}",
                  annotation_font_color=COLORS["danger"])
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=280,
        title=dict(text="Component Scores — Last 90 Days",
                   font=dict(color="#f1f5f9", size=13, family="Syne")),
        legend=dict(orientation="h", y=1.12, font_color="#94a3b8"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fin-header" style="font-size:1.5rem">🛡 FinSentinel</div>', unsafe_allow_html=True)
    st.markdown('<div class="fin-sub">autonomous anomaly detection</div>', unsafe_allow_html=True)
    st.divider()

    ticker = st.selectbox("TICKER", TICKERS, index=0)
    st.markdown("---")
    show_portfolio = st.checkbox("Portfolio Overview", value=True)
    show_agent     = st.checkbox("AI Agent Chat", value=True)
    st.markdown("---")
    # Read from Streamlit Cloud secrets first, fallback to manual input
    api_key = st.secrets.get("XAI_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = st.text_input("xAI / Groq API Key", type="password",
                                placeholder="gsk_...")
        st.caption("Or set XAI_API_KEY in Streamlit secrets")
    else:
        st.success("🔑 API key loaded from secrets", icon="✅")
    st.markdown("---")
    st.caption("Data: Yahoo Finance\nModel: IF + LSTM-AE + DQN\nAgent: xAI Grok")

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="fin-header">FinSentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="fin-sub">⚡ ML · DL · RL · Agentic AI — Real-Time Stock Anomaly Detection</div>',
            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load selected ticker (always)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Running FinSentinel pipeline on {ticker}…"):
    data = run_pipeline(ticker)

cur_score = float(data["scores"].iloc[-1])
thr       = float(data["thr"])
rl, badge, rc = risk_level(cur_score, thr)
n_anom    = len(data["anomalies"])
anom_rate = n_anom / len(data["scores"]) * 100

# ── KPI Row ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Live Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card" style="--accent:{rc}">
        <div class="metric-label">Current Score</div>
        <div class="metric-value">{cur_score:.4f}</div>
        <div class="metric-delta" style="color:{rc}">threshold: {thr:.4f}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card" style="--accent:{rc}">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value" style="font-size:1.4rem">{rl}</div>
        <div class="metric-delta"><span class="badge {badge}">{rl}</span></div>
    </div>""", unsafe_allow_html=True)

with c3:
    latest_ret = float(data["feat"]["log_returns"].iloc[-1]) * 100
    ret_color  = COLORS["success"] if latest_ret >= 0 else COLORS["danger"]
    st.markdown(f"""
    <div class="metric-card" style="--accent:{ret_color}">
        <div class="metric-label">Latest Return</div>
        <div class="metric-value" style="color:{ret_color}">{latest_ret:+.2f}%</div>
        <div class="metric-delta" style="color:#475569">1-day log return</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card" style="--accent:{COLORS['primary']}">
        <div class="metric-label">Anomalies Detected</div>
        <div class="metric-value">{n_anom}</div>
        <div class="metric-delta" style="color:#475569">{anom_rate:.1f}% of history</div>
    </div>""", unsafe_allow_html=True)

# ── Alert box ─────────────────────────────────────────────────────────────────
if rl in ("CRITICAL", "HIGH"):
    st.markdown(f"""
    <div class="alert-box">
        ⚠️ <strong style="color:#ef4444">ANOMALY ALERT</strong> —
        {ticker} is showing elevated risk at score <strong>{cur_score:.4f}</strong>
        ({((cur_score/thr)-1)*100:.1f}% above threshold). RL agent recommends: <strong>ALERT</strong>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-box-safe">
        ✅ <strong style="color:#22c55e">NORMAL</strong> —
        {ticker} within expected parameters. Score {cur_score:.4f} ({((1-(cur_score/thr))*100):.1f}% below threshold).
        RL agent recommends: <strong>MONITOR</strong>
    </div>""", unsafe_allow_html=True)

# ── Main chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Price & Anomaly Dashboard</div>', unsafe_allow_html=True)
st.plotly_chart(price_chart(data, ticker), use_container_width=True)

# ── Score breakdown ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Score Breakdown — Last 90 Days</div>', unsafe_allow_html=True)
st.plotly_chart(score_history_chart(data, ticker), use_container_width=True)

# ── Technical indicators table ────────────────────────────────────────────────
st.markdown('<div class="section-title">Technical Indicators (Latest)</div>', unsafe_allow_html=True)
feat_last = data["feat"].iloc[-1]
rsi_val   = float(feat_last["rsi_norm"]) * 50 + 50
ind_df = pd.DataFrame({
    "Indicator": ["RSI (14)", "Vol Ratio", "Bollinger Position", "MACD (norm)", "Volatility 20d", "Return 1d"],
    "Value":     [f"{rsi_val:.1f}", f"{feat_last['volume_ratio']:.2f}x",
                  f"{feat_last['bb_pos']:.3f}", f"{feat_last['macd_norm']:.5f}",
                  f"{feat_last['volatility_20']:.5f}", f"{feat_last['log_returns']*100:.2f}%"],
    "Signal":    [
        "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral",
        "Elevated ⚠" if feat_last["volume_ratio"] > 2 else "Normal",
        "Upper Band" if feat_last["bb_pos"] > 0.8 else "Lower Band" if feat_last["bb_pos"] < -0.8 else "Mid",
        "Bullish" if feat_last["macd_norm"] > 0 else "Bearish",
        "High Vol" if feat_last["volatility_20"] > 0.025 else "Low Vol",
        "Up" if feat_last["log_returns"] > 0 else "Down",
    ]
}).set_index("Indicator")
st.dataframe(ind_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Overview
# ──────────────────────────────────────────────────────────────────────────────
if show_portfolio:
    st.markdown("---")
    st.markdown('<div class="section-title">Portfolio Overview — All Tickers</div>', unsafe_allow_html=True)

    with st.spinner("Loading all tickers…"):
        port_results = {}
        prog = st.progress(0)
        for i, t in enumerate(TICKERS):
            port_results[t] = run_pipeline(t)
            prog.progress((i + 1) / len(TICKERS))
        prog.empty()

    st.plotly_chart(portfolio_chart(port_results), use_container_width=True)

    # Summary table
    rows = []
    for t, res in port_results.items():
        sc  = float(res["scores"].iloc[-1])
        thr = float(res["thr"])
        rl_t, _, _ = risk_level(sc, thr)
        rows.append({
            "Ticker": t,
            "Score":  round(sc, 4),
            "Threshold": round(thr, 4),
            "Risk":   rl_t,
            "Anomalies": len(res["anomalies"]),
            "Anom Rate": f"{len(res['anomalies'])/len(res['scores'])*100:.1f}%",
        })
    port_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(port_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Agent Chat
# ──────────────────────────────────────────────────────────────────────────────
if show_agent:
    st.markdown("---")
    st.markdown('<div class="section-title">AI Agent — Analyst Chat</div>', unsafe_allow_html=True)

    if not api_key:
        st.info("⬅ Enter your xAI API key in the sidebar to enable the AI Agent.")
    else:
        from openai import OpenAI

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        AGENT_TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "get_anomaly_score",
                    "description": "Get current anomaly score and risk level for a stock.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_technical_indicators",
                    "description": "Get RSI, volume ratio, Bollinger Band position for a stock.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_portfolio",
                    "description": "Rank all portfolio stocks by anomaly risk.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
        ]

        def run_tool(name, inp):
            if name == "get_anomaly_score":
                t   = inp["ticker"].upper()
                if t not in port_results:
                    return {"error": f"{t} not loaded"}
                res = port_results[t]
                sc  = float(res["scores"].iloc[-1])
                thr = float(res["thr"])
                rl_t, _, _ = risk_level(sc, thr)
                return {"ticker": t, "score": round(sc, 4),
                        "threshold": round(thr, 4),
                        "risk_level": rl_t,
                        "is_anomalous": sc > thr}

            elif name == "get_technical_indicators":
                t = inp["ticker"].upper()
                if t not in port_results:
                    return {"error": f"{t} not loaded"}
                fl = port_results[t]["feat"].iloc[-1]
                return {
                    "ticker":    t,
                    "RSI":       round(float(fl["rsi_norm"]) * 50 + 50, 1),
                    "vol_ratio": round(float(fl["volume_ratio"]), 2),
                    "bb_pos":    round(float(fl["bb_pos"]), 3),
                    "macd_norm": round(float(fl["macd_norm"]), 6),
                }

            elif name == "compare_portfolio":
                out = []
                for t, res in port_results.items():
                    sc  = float(res["scores"].iloc[-1])
                    thr = float(res["thr"])
                    rl_t, _, _ = risk_level(sc, thr)
                    out.append({"ticker": t, "score": round(sc, 4), "risk": rl_t})
                out.sort(key=lambda x: x["score"], reverse=True)
                return {"ranked": out, "highest_risk": out[0]["ticker"]}

            return {"error": "unknown tool"}

        SYSTEM = """You are FinSentinel AI, an elite financial analyst with access to a live ML+DL+RL anomaly detection pipeline.
Always call tools before answering. Use specific numbers. Be concise and direct."""

        if prompt := st.chat_input("Ask the agent… e.g. 'Which stock is most at risk right now?'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent thinking…"):
                    client  = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                    history = [{"role": "system", "content": SYSTEM}] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]

                    for _ in range(8):
                        resp = client.chat.completions.create(
                            model="grok-3",
                            max_tokens=1000,
                            messages=history,
                            tools=AGENT_TOOLS,
                            tool_choice="auto"
                        )
                        msg = resp.choices[0].message
                        history.append({
                            "role": "assistant",
                            "content": msg.content,
                            "tool_calls": msg.tool_calls
                        })

                        if resp.choices[0].finish_reason == "stop":
                            final = msg.content or ""
                            st.markdown(final)
                            st.session_state.messages.append({"role": "assistant", "content": final})
                            break

                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                args = json.loads(tc.function.arguments)
                                res  = run_tool(tc.function.name, args)
                                history.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": json.dumps(res)
                                })

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.65rem; letter-spacing:0.1em">
FINSENTINEL v1.0 · THAPAR UNIVERSITY · DATA: YAHOO FINANCE · MODEL: IF + LSTM-AE + DQN · AGENT: CLAUDE
</div>
""", unsafe_allow_html=True)
