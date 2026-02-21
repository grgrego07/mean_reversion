import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# --- PALETTE DEFINITION ---
MAIN_TEAL = "#00d4aa"
DARK_BG = "#0e1117"
ACCENT_RED = "#ff4b4b"
CHART_COLORS = ["#00d4aa", "#008a73", "#004d40", "#7ef4da", "#b2fcf0"]

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Institutional Quant Suite", layout="wide")

# --- FORCED WHITE SIDEBAR & DARK CSS ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {DARK_BG}; color: #ffffff; }}
    section[data-testid="stSidebar"] {{ 
        background-color: #161b22 !important; 
        border-right: 1px solid #232a35; 
    }}

    /* Force ALL Sidebar Text to White (Headers, Labels, Radio Options) */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p,
    [data-testid="stWidgetLabel"] p {{
        color: white !important;
        font-weight: 500 !important;
    }}

    /* Force Slider Range Numbers (0.10, 0.90, 10, 50) to White */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"], [data-testid="stSliderTick"] {{
        color: white !important;
    }}

    /* Force Metric Values to White */
    [data-testid="stMetricValue"] {{ color: #ffffff !important; font-weight: bold !important; }}
    [data-testid="stMetricLabel"] {{ color: #8b949e !important; }}

    /* Title and Header colors */
    h1, h2, h3 {{ color: {MAIN_TEAL} !important; font-family: 'Inter', sans-serif; }}

    /* Button Styling */
    .stButton>button {{ 
        background-color: {MAIN_TEAL} !important; 
        color: {DARK_BG} !important; 
        font-weight: bold; width: 100%; border: none;
    }}
</style>
""", unsafe_allow_html=True)

st.title("📊 Institutional Regime & Persistence Terminal")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Parameters")
    asset_type = st.radio("Asset Class", ["Commodities", "Currencies"])
    ticker_list = {
        "Commodities": {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Copper": "HG=F", "Corn": "ZC=F"},
        "Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X"}
    }[asset_type]

    selected_label = st.selectbox("Select Asset", list(ticker_list.keys()))
    ticker = ticker_list[selected_label]

    period_options = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    selected_period = st.selectbox("Lookback Horizon", list(period_options.keys()), index=1)

    st.markdown("---")
    st.header("Model Tuning")
    d_val = st.slider("FracDiff 'd' (Memory)", 0.1, 0.9, 0.4, 0.1)
    z_win = st.slider("Z-Score Window", 10, 50, 20)
    run_button = st.button("Execute Quantitative Analysis")


# --- QUANT FUNCTIONS ---

def fractional_diff(series, d, threshold=1e-2):
    val = series.values.astype(float)
    weights = [1.0]
    for k in range(1, len(val)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold: break
        weights.append(w)
    weights = np.array(weights[::-1]).reshape(-1, 1)
    res = [np.dot(weights.T, val[i - len(weights):i])[0] for i in range(len(weights), len(val))]
    return pd.Series(res, index=series.index[len(weights):])


def robust_zscore(series, window=20):
    med = series.rolling(window).median()
    mad = series.rolling(window).apply(lambda x: np.median(np.abs(x - np.median(x))))
    return (series - med) / (1.4826 * mad + 1e-6)


def ou_zscore(series, look=20):
    mu = series.rolling(look).mean()
    sigma = series.rolling(look).std(ddof=0)
    return (series - mu) / (sigma + 1e-9)


def rolling_hurst(series, win=30):
    def get_h(v):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(v[l:], v[:-l]))) for l in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2

    return series.rolling(win).apply(lambda x: get_h(x) if len(x) == win else 0.5)


def _compute_segments(index, mask_bool):
    segs = []
    start = None
    for i, flag in enumerate(mask_bool):
        if flag and start is None: start = index[i]
        if (not flag) and (start is not None):
            segs.append((start, index[i]))
            start = None
    if start is not None: segs.append((start, index[-1]))
    return segs


# --- EXECUTION ---

if run_button:
    with st.spinner(f"Analyzing {selected_label}..."):
        data = yf.download(ticker, period=period_options[selected_period])
        s = data['Close'].dropna().astype(float) if not isinstance(data.columns, pd.MultiIndex) else data['Close'][
            ticker].dropna().astype(float)
        ret = np.log(s / s.shift(1)).dropna()

        s_frac = fractional_diff(np.log(s), d=d_val).dropna()
        s_input = ((s_frac - s_frac.mean()) / s_frac.std())

        try:
            model = MarkovRegression(s_input, k_regimes=2, switching_variance=True)
            res = model.fit(method='bfgs', maxiter=100, disp=False)
            h_idx = 1 if res.params.get('sigma2[1]', 0) > res.params.get('sigma2[0]', 0) else 0
            probs = res.smoothed_marginal_probabilities[h_idx].reindex(s.index).ffill().fillna(0)
        except:
            probs = pd.Series(0, index=s.index)

        z_rob = robust_zscore(s, window=z_win)
        z_ou = ou_zscore(s, look=z_win)
        hurst = rolling_hurst(s)
        kelly = (1 - probs) * 0.2
        rolling_acf = ret.rolling(z_win).apply(lambda x: acf(x, nlags=1)[1] if len(x) == z_win else 0)
        acf_dist = acf(ret, nlags=20)
        reg_segs = _compute_segments(s.index, (probs.values > 0.5))

        # Metrics Panel
        c1, c2, c3 = st.columns(3)
        c1.metric("Regime Risk", f"{probs.iloc[-1]:.1%}")
        c2.metric("Market Persistence", f"{rolling_acf.iloc[-1]:.2f}")
        c3.metric("Kelly Scale", f"{kelly.iloc[-1]:.1%}")

        # --- 9-GRAPH DASHBOARD ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(9, 1, figsize=(12, 45), facecolor=DARK_BG)
        loc = mdates.AutoDateLocator()
        fmt = mdates.ConciseDateFormatter(loc)

        # 1. Price
        ax[0].plot(s.index, s, color='white', alpha=0.3)
        ax[0].scatter(s.index, s, c=probs, cmap='RdYlGn_r', s=15, zorder=3)
        ax[0].set_title(f"{selected_label} | Price & Risk Regime", color=MAIN_TEAL, loc='left')

        # 2. Robust Z-Score (Consistent Teal)
        ax[1].plot(z_rob.index, z_rob, color=MAIN_TEAL, lw=1.5)
        ax[1].axhline(2, color=ACCENT_RED, ls='--');
        ax[1].axhline(-2, color=MAIN_TEAL, ls='--')
        ax[1].set_title("Robust Z-Score (Mean Reversion)", color=MAIN_TEAL, loc='left')

        # 3. OU Z-Score (Consistent Teal)
        ax[2].plot(z_ou.index, z_ou, color=MAIN_TEAL, lw=1.5, alpha=0.8)
        ax[2].axhline(1, color='grey', ls='--');
        ax[2].axhline(-1, color='grey', ls='--')
        ax[2].set_title("OU Z-Score (Signal)", color=MAIN_TEAL, loc='left')

        # 4. Hurst & Probability
        ax[3].plot(probs.index, probs, color=ACCENT_RED, alpha=0.4)
        ax3_t = ax[3].twinx()
        ax3_t.plot(hurst.index, hurst, color=MAIN_TEAL, lw=1.5)
        ax[3].set_title("HMM Risk vs. Rolling Hurst", color=MAIN_TEAL, loc='left')

        # 5. Kelly Allocation
        ax[4].fill_between(kelly.index, 0, kelly, color=MAIN_TEAL, alpha=0.4)
        ax[4].set_title("Dynamic Kelly Allocation", color=MAIN_TEAL, loc='left')

        # 6. Volatility
        ax[5].plot(ret.rolling(20).std() * np.sqrt(252), color=MAIN_TEAL, lw=1.2)
        ax[5].set_title("Rolling Annualized Volatility", color=MAIN_TEAL, loc='left')

        # 7. Market Persistence (Consistent Teal Bar)
        ax[6].bar(rolling_acf.index, rolling_acf, color=MAIN_TEAL, alpha=0.6, width=1.0)
        ax[6].axhline(0, color='white', lw=0.5)
        ax[6].set_title("Market Memory (Persistence)", color=MAIN_TEAL, loc='left')

        # 8. ACF Distribution
        ax[7].bar(range(len(acf_dist)), acf_dist, color=MAIN_TEAL, alpha=0.8)
        ax[7].set_title("Static ACF (Lag Memory)", color=MAIN_TEAL, loc='left')

        # 9. Returns Density
        ax[8].hist(ret, bins=30, color=MAIN_TEAL, alpha=0.4, density=True)
        ax[8].set_title("Returns Density", color=MAIN_TEAL, loc='left')

        # Final Formatting
        for i, a in enumerate(ax):
            a.set_facecolor(DARK_BG)
            a.grid(True, alpha=0.05, color='white')
            for spine in a.spines.values(): spine.set_edgecolor('#232a35')
            if i < 7:  # Temporal charts
                for (start, end) in reg_segs:
                    a.axvspan(start, end, color=ACCENT_RED, alpha=0.08, lw=0)
                a.xaxis.set_major_locator(loc);
                a.xaxis.set_major_formatter(fmt)
                a.tick_params(labelbottom=True, colors='#8b949e', labelsize=9)
            else:
                a.tick_params(colors='#8b949e', labelsize=9)

        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Select asset parameters and click Execute.")