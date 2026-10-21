

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
import math

TZ = ZoneInfo("Asia/Ho_Chi_Minh")

st.set_page_config(
    page_title="duancophieu01 - Price Levels",
    page_icon="📈",
    layout="wide",
)

STOCKS = {
    "FPT": "FPT.VN",
    "VCB": "VCB.VN",
    "HPG": "HPG.VN",
    "VNM": "VNM.VN",
    "MWG": "MWG.VN",
    "SSI": "SSI.VN",
    "GAS": "GAS.VN",
    "VIC": "VIC.VN",
    "BID": "BID.VN",
    "CTG": "CTG.VN",
    "MBB": "MBB.VN",
}

EXCHANGE_BANDS = {
    "HOSE": 0.07,
    "HNX": 0.10,
    "UPCoM": 0.15,
}


@st.cache_data(ttl=300, show_spinner=False)
def load_history(ticker, period="1y"):
    """Lấy dữ liệu lịch sử từ Yahoo Finance; không tạo dữ liệu thay thế."""
    stock = yf.Ticker(ticker)
    df = stock.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    required = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required + (["volume"] if "volume" in df.columns else [])]
    df = df.dropna(subset=required)
    df = df.sort_index()

    return df


def tick_size(price, exchange):
    """
    Tick size theo đơn vị giá niêm yết mà nguồn dữ liệu trả về.
    Yahoo Finance có thể biểu diễn giá cổ phiếu Việt Nam theo nghìn đồng.
    Cần đối chiếu đơn vị và quy tắc của sở giao dịch trước khi sử dụng.
    """
    if exchange == "HOSE":
        if price < 10:
            return 0.01
        elif price < 50:
            return 0.05
        return 0.10

    # HNX và UPCoM thường dùng bước giá 100 đồng.
    return 0.10


def round_to_tick(price, tick, mode):
    if not np.isfinite(price) or tick <= 0:
        return np.nan

    units = price / tick

    if mode == "down":
        return math.floor(units + 1e-9) * tick

    if mode == "up":
        return math.ceil(units - 1e-9) * tick

    return round(units) * tick


def estimate_price_limits(reference, exchange):
    """
    Ước tính biên độ trần/sàn từ giá tham chiếu.
    Đây không phải dữ liệu trần/sàn chính thức từ sở giao dịch.
    """
    if not np.isfinite(reference) or reference <= 0:
        return np.nan, np.nan

    band = EXCHANGE_BANDS[exchange]
    tick = tick_size(reference, exchange)

    ceiling_raw = reference * (1 + band)
    floor_raw = reference * (1 - band)

    ceiling = round_to_tick(ceiling_raw, tick, "down")
    floor = round_to_tick(floor_raw, tick, "up")

    return ceiling, floor


def calculate_indicators(df):
    result = df.copy()

    result["MA20"] = result["close"].rolling(20).mean()
    result["MA50"] = result["close"].rolling(50).mean()

    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["ATR14"] = true_range.rolling(14).mean()

    # Mức tham khảo lấy từ 20 phiên đã có dữ liệu.
    result["Support20"] = result["low"].rolling(20).min()
    result["Resistance20"] = result["high"].rolling(20).max()

    return result


def fmt(value, digits=2):
    if value is None or pd.isna(value):
        return "Chưa đủ dữ liệu"
    return f"{value:,.{digits}f}"


st.title("📈 duancophieu01")
st.subheader("Phân tích mức giá theo dữ liệu thị trường")
st.caption(
    "Nguồn dữ liệu: Yahoo Finance | "
    "Giá lịch sử có thể trễ và không được bảo đảm là giá khớp lệnh thời gian thực."
)

with st.sidebar:
    st.header("Thiết lập")

    symbol = st.selectbox(
        "Chọn mã cổ phiếu",
        list(STOCKS.keys()),
        index=list(STOCKS.keys()).index("FPT"),
    )

    exchange = st.selectbox(
        "Sàn giao dịch",
        ["HOSE", "HNX", "UPCoM"],
        index=0,
        help="Chọn đúng sàn niêm yết của mã để ước tính biên độ giá.",
    )

    period = st.selectbox(
        "Khoảng lịch sử",
        ["6mo", "1y", "2y", "5y"],
        index=1,
    )

    if st.button("Làm mới dữ liệu"):
        load_history.clear()
        st.rerun()

ticker = STOCKS[symbol]

with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    raw = load_history(ticker, period)

if raw.empty:
    st.error(
        f"Không lấy được dữ liệu cho {symbol} ({ticker}). "
        "Không có giá giả được sử dụng. Hãy thử lại hoặc kiểm tra mã và kết nối."
    )
    st.stop()

df = calculate_indicators(raw)
last = df.iloc[-1]
previous = df.iloc[-2] if len(df) >= 2 else None

latest_close = float(last["close"])
previous_close = float(previous["close"]) if previous is not None else np.nan

change = (
    latest_close - previous_close
    if np.isfinite(previous_close)
    else np.nan
)

change_pct = (
    change / previous_close * 100
    if np.isfinite(change) and previous_close != 0
    else np.nan
)

# Giá tham chiếu để ước tính biên độ là giá đóng cửa phiên trước.
reference = previous_close
ceiling, floor = estimate_price_limits(reference, exchange)

data_timestamp = last.name
try:
    if data_timestamp.tzinfo is None:
        data_timestamp = data_timestamp.tz_localize(TZ)
    else:
        data_timestamp = data_timestamp.tz_convert(TZ)
except Exception:
    pass

st.success("Đã nhận dữ liệu lịch sử từ Yahoo Finance")

st.caption(f"Mã: {symbol} | Nguồn: Yahoo Finance")
st.caption(f"Phiên dữ liệu cuối cùng: {data_timestamp}")
st.caption(
    "Đây là thời điểm của bản ghi lịch sử cuối cùng, "
    "không phải thời điểm xác nhận giá đang khớp lệnh trực tiếp."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Giá đóng cửa gần nhất", fmt(latest_close))
c2.metric(
    "Thay đổi so phiên trước",
    fmt(change),
    delta=f"{change_pct:+.2f}%" if np.isfinite(change_pct) else None,
)
c3.metric("Cao nhất phiên", fmt(last["high"]))
c4.metric("Thấp nhất phiên", fmt(last["low"]))

st.divider()
st.header("Các mức giá tham khảo")

support = last["Support20"]
resistance = last["Resistance20"]
ma20 = last["MA20"]
ma50 = last["MA50"]
atr = last["ATR14"]

# Các mốc theo biên độ ATR chỉ là vùng biến động tham khảo.
atr_lower = latest_close - atr if pd.notna(atr) else np.nan
atr_upper = latest_close + atr if pd.notna(atr) else np.nan

m1, m2, m3 = st.columns(3)
m1.metric("Hỗ trợ tham khảo · đáy 20 phiên", fmt(support))
m2.metric("Kháng cự tham khảo · đỉnh 20 phiên", fmt(resistance))
m3.metric("Biên độ ATR14", fmt(atr))

m4, m5, m6 = st.columns(3)
m4.metric("MA20", fmt(ma20))
m5.metric("MA50", fmt(ma50))
m6.metric("Giá tham chiếu ước tính", fmt(reference))

m7, m8 = st.columns(2)
m7.metric("Giá trần ước tính", fmt(ceiling))
m8.metric("Giá sàn ước tính", fmt(floor))

st.caption(
    f"Sàn đã chọn: {exchange} | Biên độ giả định: "
    f"±{EXCHANGE_BANDS[exchange] * 100:.0f}% so với giá tham chiếu."
)

st.warning(
    "Giá trần/sàn ở đây chỉ là ước tính theo biên độ và bước giá giả định. "
    "Giá chính thức có thể khác do quy tắc làm tròn, điều chỉnh giá, "
    "ngày giao dịch không hưởng quyền hoặc quy định của sở. "
    "Không dùng các mức ước tính này để gửi lệnh giao dịch."
)

st.subheader("Biểu đồ giá và các mức tham khảo")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=symbol,
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA20"],
        mode="lines",
        name="MA20",
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA50"],
        mode="lines",
        name="MA50",
    )
)

if pd.notna(support):
    fig.add_hline(
        y=support,
        line_dash="dash",
        annotation_text="Hỗ trợ 20 phiên",
    )

if pd.notna(resistance):
    fig.add_hline(
        y=resistance,
        line_dash="dash",
        annotation_text="Kháng cự 20 phiên",
    )

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    xaxis_title="Ngày giao dịch",
    yaxis_title="Giá theo Yahoo Finance",
    legend=dict(orientation="h"),
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Bảng dữ liệu lịch sử")
st.dataframe(
    df.sort_index(ascending=False),
    use_container_width=True,
)

csv = df.to_csv(index=True).encode("utf-8-sig")
st.download_button(
    "Tải dữ liệu CSV",
    data=csv,
    file_name=f"{symbol}_price_levels.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Các mức hỗ trợ/kháng cự được tính từ dữ liệu quá khứ, "
    "không bảo đảm giá sẽ đảo chiều tại các mức này. "
    "Ứng dụng cung cấp thông tin tham khảo, không phải khuyến nghị đầu tư."
)
'''

st.code("pip install streamlit yfinance pandas numpy plotly", language="bash")
