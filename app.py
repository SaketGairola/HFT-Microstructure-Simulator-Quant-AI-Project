import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.write("🚀 App started successfully")

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="HFT Microstructure Sim | Quant Desk", page_icon="⚡", layout="wide")
st.title("⚡ LOB Synthesizer & Alpha Execution Simulator")
st.caption("Live NSE Proxy | Synthetic Level 2 Data | Taker + Alpha Strategy | #MarketMicrostructure")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ 1. Universe & Timeframe")
    ticker = st.selectbox("NSE Proxy Ticker", ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "^NSEI"])
    col1, col2 = st.columns(2)
    interval = col1.selectbox("Interval", ["1m", "5m"])
    days = col2.slider("Period (Days)", 1, 7, 5)
    
    st.markdown("---")
    st.header("📈 2. Alpha Filters (The Brains)")
    st.caption("Only execute if the broader trend aligns.")
    use_trend = st.checkbox("Require EMA Alignment (20 > 50)", value=True)
    use_rsi = st.checkbox("Require RSI Filter (Buy > 50, Sell < 50)", value=True)

    st.markdown("---")
    st.header("🎛️ 3. Execution Params (The Brawn)")
    st.caption("Taker logic: Cross spread when tight & LOB is favorable.")
    spread_threshold = st.slider("Max Spread Entry (bps)", 1, 30, 10)
    imb_threshold = st.slider("Imbalance Threshold", 0.1, 0.9, 0.3, 0.05)
    
    st.markdown("---")
    st.header("🛡️ 4. Risk & Inventory")
    max_inv = st.number_input("Max Inventory (Shares)", value=1000, step=100)
    trade_qty = st.number_input("Order Size", value=100, step=10)
    cost_bps = st.number_input("Fee per Trade (bps)", value=1.0, step=0.5) / 10000

    st.markdown("---")
    if st.button("🔄 Run Simulation", use_container_width=True):
        st.cache_data.clear()

# --- 3. DATA ENGINE & SYNTHETIC LOB ---
@st.cache_data(show_spinner="Synthesizing Level 2 Data & Computing Alpha...")
def load_and_synthesize(tkr, d, inv):
    df = yf.download(tkr, period=f"{d}d", interval=inv, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return df, None, None

    df['Mid'] = df['Close']
    df['Returns'] = df['Mid'].pct_change().fillna(0)
    df['Vol_5'] = df['Returns'].rolling(window=5).std().fillna(0)

    # --- Alpha Calculations ---
    df['EMA_20'] = df['Mid'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Mid'].ewm(span=50, adjust=False).mean()
    
    delta = df['Mid'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) 

    # --- LOB Synthesis ---
    tick_size = df['Mid'].iloc[0] * 0.0001 
    base_spread = tick_size * 2

    vol_scaler = df['Vol_5'] / (df['Vol_5'].mean() + 1e-8)
    np.random.seed(42)
    df['Spread'] = base_spread * (1 + 0.3 * vol_scaler + np.random.uniform(-0.2, 0.8, len(df)))
    df['Spread'] = np.maximum(df['Spread'], tick_size)
    df['Spread_bps'] = (df['Spread'] / df['Mid']) * 10000

    b_depths = np.random.lognormal(mean=np.log(1000), sigma=0.6, size=(len(df), 5))
    a_depths = np.random.lognormal(mean=np.log(1000), sigma=0.6, size=(len(df), 5))
    
    df['Bid_Depth'] = b_depths.sum(axis=1)
    df['Ask_Depth'] = a_depths.sum(axis=1)
    df['Imbalance'] = (df['Bid_Depth'] - df['Ask_Depth']) / (df['Bid_Depth'] + df['Ask_Depth'] + 1e-8)
    
    return df, b_depths, a_depths

df, bids, asks = load_and_synthesize(ticker, days, interval)

st.write("Data loaded:", df.shape)

if df.empty or bids is None or asks is None:
    st.warning("⚠️ No data from Yahoo. Using dummy data for demo.")

    import pandas as pd
    import numpy as np

    index = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1min")
    df = pd.DataFrame({
        "Mid": np.linspace(100, 110, 100),
        "Spread": np.random.uniform(0.01, 0.05, 100),
        "Spread_bps": np.random.uniform(5, 15, 100),
        "Imbalance": np.random.uniform(-0.5, 0.5, 100),
        "EMA_20": np.linspace(100, 110, 100),
        "EMA_50": np.linspace(100, 108, 100),
        "RSI": np.random.uniform(40, 60, 100),
    }, index=index)

    bids = np.random.rand(100, 5) * 1000
    asks = np.random.rand(100, 5) * 1000

# --- 4. BACKTESTING LOGIC (ALPHA + LIQUIDITY TAKER) ---
st.write("Running backtest...")
spread_thresh_dec = spread_threshold / 10000.0

mids = df['Mid'].values
spreads = df['Spread'].values
imbs = df['Imbalance'].values
spread_bps = df['Spread_bps'].values
ema20 = df['EMA_20'].values
ema50 = df['EMA_50'].values
rsi = df['RSI'].values

cash = 0.0
inventory = 0
positions = np.zeros(len(df))
pnl_curve = np.zeros(len(df))
trades = []

for i in range(len(df)):
    mid = mids[i]
    spread = spreads[i]
    imb = imbs[i]
    
    ask = mid + spread/2
    bid = mid - spread/2
    is_tight = (spread / mid) <= spread_thresh_dec
    
    trend_up = ema20[i] > ema50[i]
    trend_down = ema20[i] < ema50[i]
    rsi_bullish = rsi[i] > 50
    rsi_bearish = rsi[i] < 50
    
    micro_buy = is_tight and imb > imb_threshold and inventory < max_inv
    micro_sell = is_tight and imb < -imb_threshold and inventory > -max_inv
    
    if use_trend:
        micro_buy = micro_buy and trend_up
        micro_sell = micro_sell and trend_down
    if use_rsi:
        micro_buy = micro_buy and rsi_bullish
        micro_sell = micro_sell and rsi_bearish

    if micro_buy:
        fee = ask * trade_qty * cost_bps
        cash -= (ask * trade_qty + fee)
        inventory += trade_qty
        trades.append({'Index': df.index[i], 'Side': 'BUY', 'Price': ask, 'Size': trade_qty, 'Spread_bps': spread_bps[i], 'Fee': fee})
        
    elif micro_sell:
        fee = bid * trade_qty * cost_bps
        cash += (bid * trade_qty - fee)
        inventory -= trade_qty
        trades.append({'Index': df.index[i], 'Side': 'SELL', 'Price': bid, 'Size': trade_qty, 'Spread_bps': spread_bps[i], 'Fee': fee})
        
    positions[i] = inventory
    pnl_curve[i] = cash + (inventory * mid)

df['Inventory'] = positions
df['PnL'] = pnl_curve
trades_df = pd.DataFrame(trades).set_index('Index') if trades else pd.DataFrame(columns=['Side', 'Price', 'Size', 'Spread_bps', 'Fee'])

# --- 5. HERO KPIs ---
total_pnl = df['PnL'].iloc[-1]
fees_paid = trades_df['Fee'].sum() if not trades_df.empty else 0.0
gross_pnl = total_pnl + fees_paid
returns = df['PnL'].diff().fillna(0)
sharpe = np.sqrt(252 * (len(df)/max(1, days))) * (returns.mean() / (returns.std() + 1e-8)) if returns.std() > 0 else 0
turnover = (trades_df['Size'] * trades_df['Price']).sum() if not trades_df.empty else 0.0
avg_spd_paid = trades_df['Spread_bps'].mean() if not trades_df.empty else 0.0

st.markdown("### 📊 Alpha-Adjusted Performance")
c1, c2, c3, c4 = st.columns(4)

# FIX: Added delta_color="off" so negative gross PnL stays gray instead of highlighting green
c1.metric("Total Net PnL", f"₹ {total_pnl:,.2f}", f"Gross: ₹ {gross_pnl:,.2f}", delta_color="off")
c2.metric("Sharpe Ratio (Sim)", f"{sharpe:.2f}")
c3.metric("Avg Spread Paid", f"{avg_spd_paid:.2f} bps")
c4.metric("Turnover Traded", f"₹ {turnover:,.0f}", f"{len(trades_df)} Executions")

# --- 6. MASTER PLOTLY DASHBOARD ---
st.markdown("### 🖥️ Execution Analytics")

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Mid'], mode='lines', name='Midprice', line=dict(color='#555', width=1)))
    if use_trend:
        fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='cyan', width=1, dash='dot')))
        fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='magenta', width=1, dash='dot')))
        
    if not trades_df.empty:
        buys = trades_df[trades_df['Side'] == 'BUY']
        sells = trades_df[trades_df['Side'] == 'SELL']
        fig1.add_trace(go.Scatter(x=buys.index, y=buys['Price'], mode='markers', name='Taker Buy', marker=dict(color='lime', symbol='triangle-up', size=8)))
        fig1.add_trace(go.Scatter(x=sells.index, y=sells['Price'], mode='markers', name='Taker Sell', marker=dict(color='red', symbol='triangle-down', size=8)))
    fig1.update_layout(title="Execution Map with Alpha Overlays", template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with r1c2:
    lookback = min(150, len(df))
    heat_z = np.zeros((10, lookback))
    for i, idx in enumerate(range(len(df)-lookback, len(df))):
        heat_z[0:5, i] = asks[idx][::-1]  
        heat_z[5:10, i] = bids[idx]       
    
    fig2 = go.Figure(data=go.Heatmap(
        z=heat_z, x=df.index[-lookback:],
        y=["A5", "A4", "A3", "A2", "A1", "B1", "B2", "B3", "B4", "B5"],
        colorscale="Cividis", showscale=False
    ))
    fig2.update_layout(title=f"LOB Depth Heatmap (Last {lookback} bars)", template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with r2c1:
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=df.index, y=df['PnL'], mode='lines', name='Net PnL', line=dict(color='gold', width=2)), secondary_y=False)
    fig3.add_trace(go.Scatter(x=df.index, y=df['Inventory'], mode='lines', name='Inventory', line=dict(color='cyan', width=1, dash='dot')), secondary_y=True)
    fig3.update_layout(title="Cumulative PnL & Inventory State", template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    fig3.update_yaxes(title_text="PnL (₹)", secondary_y=False)
    fig3.update_yaxes(title_text="Position", secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Spread Dist (bps)", "Avg LOB Depth"))
    fig4.add_trace(go.Histogram(x=df['Spread_bps'], nbinsx=40, marker_color='magenta', name="Spread"), row=1, col=1)
    
    avg_bids = np.mean(bids, axis=0)
    avg_asks = np.mean(asks, axis=0)
    levels_x = ['L1', 'L2', 'L3', 'L4', 'L5']
    fig4.add_trace(go.Bar(x=levels_x, y=avg_bids, name='Bids', marker_color='#2e8b57'), row=1, col=2)
    fig4.add_trace(go.Bar(x=levels_x, y=avg_asks, name='Asks', marker_color='#d62728'), row=1, col=2)
    
    fig4.update_layout(title="Market Microstructure Analytics", template="plotly_dark", height=350, barmode='group', margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

# --- 7. TRADE LOG & STYLED DATAFRAME ---
st.markdown("### 📜 Recent Executions & Trade Log")

if not trades_df.empty:
    log_df = trades_df.tail(15).copy()
    
    def color_side(val):
        color = '#2e8b57' if val == 'BUY' else '#d62728'
        return f'color: {color}; font-weight: bold'

    # FIX: Added try/except to handle Colab's older Pandas environments safely
    try:
        styled_log = log_df.style.format({
            'Price': '₹{:,.2f}',
            'Size': '{:.0f}',
            'Spread_bps': '{:.2f} bps',
            'Fee': '₹{:.2f}'
        }).map(color_side, subset=['Side'])
    except AttributeError:
        styled_log = log_df.style.format({
            'Price': '₹{:,.2f}',
            'Size': '{:.0f}',
            'Spread_bps': '{:.2f} bps',
            'Fee': '₹{:.2f}'
        }).applymap(color_side, subset=['Side'])

    st.dataframe(styled_log, use_container_width=True)
    
    csv = trades_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Full Trade Log CSV", data=csv, file_name='hft_executions.csv', mime='text/csv')
else:
    st.info("No trades executed. The Alpha filters + Microstructure thresholds might be too strict. Try loosening them.")