
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import feedparser
import time
import html
from datetime import datetime, timedelta

# =========================================================
# STOCKVISION VIETNAM - STREAMLIT APP
# =========================================================

st.set_page_config(
    page_title="StockVision Vietnam",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DARK THEME -------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}
h1, h2, h3 {
    color: #f3f4f6 !important;
}
p, label, span {
    color: inherit;
}
div[data-testid="stMetric"] {
    background: #111c2e;
    border: 1px solid #27364d;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #f8fafc;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}
div[data-testid="stForm"] {
    background: #111827;
    border: 1px solid #263247;
    border-radius: 12px;
    padding: 16px;
}
.stButton button {
    border-radius: 8px;
    font-weight: 600;
}
.small-note {
    color: #94a3b8;
    font-size: 12px;
}
.market-card {
    background: #111c2e;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #27364d;
}
a {
    color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- CONSTANTS -------------------------

POPULAR = [
    "FPT", "SSI", "VNM", "VCB", "HPG",
    "VIC", "VHM", "MWG", "TCB", "MBB",
    "CTG", "ACB", "GAS", "DGC", "VND"
]

COMPANIES = {
    "FPT": ("FPT Corporation", "HOSE"),
    "SSI": ("SSI Securities Corporation", "HOSE"),
    "VNM": ("Vietnam Dairy Products JSC", "HOSE"),
    "VCB": ("Vietcombank", "HOSE"),
    "HPG": ("Hoa Phat Group", "HOSE"),
    "VIC": ("Vingroup JSC", "HOSE"),
    "VHM": ("Vinhomes JSC", "HOSE"),
    "MWG": ("Mobile World Investment Corporation", "HOSE"),
    "TCB": ("Techcombank", "HOSE"),
    "MBB": ("Military Commercial Joint Stock Bank", "HOSE"),
    "CTG": ("VietinBank", "HOSE"),
    "ACB": ("Asia Commercial Bank", "HOSE"),
    "GAS": ("PV GAS", "HOSE"),
    "DGC": ("Duc Giang Chemicals Group", "HOSE"),
    "VND": ("VNDirect Securities", "HOSE"),
    "HNX": ("HNX Index", "HNX"),
}

def company_name(symbol):
    return COMPANIES.get(
        symbol, (f"{symbol} Corporation", "Chưa xác định")
    )[0]

def exchange_name(symbol):
    return COMPANIES.get(symbol, ("", "Chưa xác định"))[1]

def fmt_price(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"

def fmt_vnd(value):
    return f"{value:,.0f} đ"

# -------------------- SESSION STATE ---------------------

def init_state():
    defaults = {
        "watchlist": ["FPT", "SSI", "VNM", "VCB"],
        "alerts": {},
        "initial_capital": 100_000_000.0,
        "cash": 100_000_000.0,
        "holdings": {},
        "transactions": [],
        "alert_history": [],
        "last_alert_check": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# -------------------- MARKET DATA -----------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol, period="1y"):
    """
    Fetch Yahoo Finance historical data.
    Vietnam tickers are commonly represented as FPT.VN.
    Yahoo Finance availability and update frequency vary.
    """
    ticker = symbol if symbol.startswith("^") else f"{symbol}.VN"

    try:
        df = yf.Ticker(ticker).history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

        if df is None or df.empty:
            return pd.DataFrame(), "unavailable"

        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]

        if "Date" not in df.columns:
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            else:
                return pd.DataFrame(), "unavailable"

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(
            None
        ) if getattr(df["Date"].dtype, "tz", None) else pd.to_datetime(
            df["Date"]
        )

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan

        df = df.dropna(subset=["Close"])
        return df, "external"

    except Exception:
        return pd.DataFrame(), "unavailable"


def make_demo_history(symbol, days=260):
    """
    Deterministic illustrative data only.
    Never represents actual market quotes.
    """
    seed = sum(ord(c) for c in symbol)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=days
    )

    base = 20 + seed % 100
    returns = rng.normal(0.0003, 0.018, days)
    closes = base * np.exp(np.cumsum(returns))

    opens = closes * (1 + rng.normal(0, 0.008, days))
    highs = np.maximum(opens, closes) * (
        1 + rng.uniform(0, 0.015, days)
    )
    lows = np.minimum(opens, closes) * (
        1 - rng.uniform(0, 0.015, days)
    )
    volumes = rng.integers(100_000, 5_000_000, days)

    return pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    })


def get_stock(symbol, allow_demo=True):
    df, source = get_history(symbol)

    if not df.empty:
        return df, "Dữ liệu lịch sử từ Yahoo Finance", False

    if allow_demo:
        return (
            make_demo_history(symbol),
            "DỮ LIỆU MINH HỌA — không phải giá thị trường",
            True
        )

    return pd.DataFrame(), "Không có dữ liệu", False


def get_last_price(symbol):
    df, source, demo = get_stock(symbol)

    if df.empty:
        return None

    close = float(df["Close"].iloc[-1])
    previous = (
        float(df["Close"].iloc[-2])
        if len(df) >= 2 else close
    )

    return {
        "symbol": symbol,
        "price": close,
        "previous": previous,
        "change": close - previous,
        "pct": (close - previous) / previous * 100
        if previous else 0,
        "volume": float(df["Volume"].iloc[-1] or 0),
        "source": source,
        "demo": demo,
        "date": df["Date"].iloc[-1],
    }


def market_status():
    return (
        "Lịch sử / có thể trễ"
        if not st.session_state.get("realtime_api")
        else "Chưa xác minh thời gian thực"
    )

# -------------------- CHARTS ----------------------------

def candle_chart(df, symbol):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol,
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444"
    ))

    fig.update_layout(
        title=f"Biểu đồ nến — {symbol}",
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        height=480,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)


def volume_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        marker_color="#3b82f6",
        name="Khối lượng"
    ))
    fig.update_layout(
        title="Khối lượng giao dịch",
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        height=260,
        margin=dict(l=10, r=10, t=45, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def performance_chart(df):
    df = df.copy()
    df["Return"] = (
        df["Close"] / df["Close"].iloc[0] - 1
    ) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Return"],
        mode="lines",
        name="Hiệu suất (%)",
        line=dict(color="#60a5fa", width=2)
    ))
    fig.update_layout(
        title="Hiệu suất giá trong kỳ (%)",
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        height=300,
        margin=dict(l=10, r=10, t=45, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MARKET TABLE ----------------------

def build_stock_table(symbols):
    rows = []

    for symbol in symbols:
        info = get_last_price(symbol)
        if info is None:
            continue

        rows.append({
            "Mã": symbol,
            "Doanh nghiệp": company_name(symbol),
            "Giá gần nhất": round(info["price"], 2),
            "Thay đổi": round(info["change"], 2),
            "% thay đổi": round(info["pct"], 2),
            "Khối lượng": int(info["volume"]),
            "Nguồn": "Minh họa" if info["demo"] else "Bên ngoài",
        })

    return pd.DataFrame(rows)


def render_stock_table(df):
    if df.empty:
        st.info("Chưa có dữ liệu để hiển thị.")
        return

    styled = df.style.map(
        lambda v: (
            "color: #22c55e" if isinstance(v, (int, float))
            and v > 0 else
            "color: #ef4444" if isinstance(v, (int, float))
            and v < 0 else ""
        ),
        subset=["Thay đổi", "% thay đổi"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# -------------------- PORTFOLIO -------------------------

def holdings_frame():
    rows = []

    for symbol, holding in st.session_state.holdings.items():
        qty = holding["qty"]
        avg = holding["avg_cost"]
        info = get_last_price(symbol)
        price = info["price"] if info else avg

        market_value = qty * price * 1000
        cost_value = qty * avg * 1000
        pnl = market_value - cost_value

        rows.append({
            "Mã": symbol,
            "Số lượng": qty,
            "Giá vốn (nghìn đồng)": avg,
            "Giá hiện tại (nghìn đồng)": price,
            "Giá trị thị trường (đ)": market_value,
            "Lãi/lỗ (đ)": pnl,
            "Lãi/lỗ (%)": pnl / cost_value * 100
            if cost_value else 0,
            "Minh họa": info["demo"] if info else True
        })

    return pd.DataFrame(rows)


def execute_trade(symbol, side, quantity, price):
    if quantity <= 0 or price <= 0:
        return False, "Số lượng và giá phải lớn hơn 0."

    # Stock prices are expressed in thousand VND.
    total = quantity * price * 1000
    holdings = st.session_state.holdings
    existing = holdings.get(
        symbol, {"qty": 0, "avg_cost": 0.0}
    )

    if side == "MUA":
        if total > st.session_state.cash:
            return False, "Không đủ tiền mặt trong danh mục mô phỏng."

        new_qty = existing["qty"] + quantity
        new_avg = (
            existing["qty"] * existing["avg_cost"]
            + quantity * price
        ) / new_qty

        holdings[symbol] = {
            "qty": new_qty,
            "avg_cost": new_avg
        }
        st.session_state.cash -= total

    elif side == "BÁN":
        if quantity > existing["qty"]:
            return False, "Số lượng bán vượt quá số cổ phiếu đang nắm giữ."

        new_qty = existing["qty"] - quantity
        st.session_state.cash += total

        if new_qty == 0:
            holdings.pop(symbol, None)
        else:
            holdings[symbol] = {
                "qty": new_qty,
                "avg_cost": existing["avg_cost"]
            }

    st.session_state.transactions.append({
        "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mã": symbol,
        "Loại": side,
        "Số lượng": quantity,
        "Giá (nghìn đồng)": price,
        "Giá trị (đ)": total
    })

    return True, "Giao dịch mô phỏng thành công."


# -------------------- FUNDAMENTALS ----------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(symbol):
    try:
        info = yf.Ticker(f"{symbol}.VN").info or {}

        def safe_num(key):
            value = info.get(key)
            try:
                value = float(value)
                return value if np.isfinite(value) else None
            except (TypeError, ValueError):
                return None

        return {
            "Tên doanh nghiệp": info.get(
                "longName", company_name(symbol)
            ),
            "Sàn": info.get(
                "exchange", exchange_name(symbol)
            ),
            "P/E": safe_num("trailingPE"),
            "P/B": safe_num("priceToBook"),
            "EPS": safe_num("trailingEps"),
            "ROE (%)": (
                safe_num("returnOnEquity") * 100
                if safe_num("returnOnEquity") is not None
                else None
            ),
            "Vốn hóa": safe_num("marketCap"),
            "Nguồn": "Yahoo Finance; có thể thiếu dữ liệu"
        }
    except Exception:
        return {
            "Tên doanh nghiệp": company_name(symbol),
            "Sàn": exchange_name(symbol),
            "P/E": None,
            "P/B": None,
            "EPS": None,
            "ROE (%)": None,
            "Vốn hóa": None,
            "Nguồn": "Không lấy được dữ liệu cơ bản"
        }


# -------------------- NEWS ------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_news(query="chứng khoán Việt Nam"):
    url = (
        "https://news.google.com/rss/search?q="
        + requests.utils.quote(query)
        + "&hl=vi&gl=VN&ceid=VN:vi"
    )

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "StockVision/1.0"}
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        articles = []
        for item in feed.entries[:15]:
            articles.append({
                "title": item.get("title", "Không có tiêu đề"),
                "link": item.get("link", ""),
                "published": item.get("published", "Không rõ thời gian"),
                "source": item.get("source", {}).get(
                    "title", "Google News"
                )
            })

        return articles

    except Exception:
        return []


# -------------------- SIDEBAR ---------------------------

with st.sidebar:
    st.markdown("# 📈 StockVision")
    st.caption("Vietnam Stock Market Analytics")
    st.divider()

    page = st.radio(
        "ĐIỀU HƯỚNG",
        [
            "Tổng quan thị trường",
            "Tra cứu cổ phiếu",
            "Danh sách theo dõi",
            "Danh mục mô phỏng",
            "Phân tích & kiến thức",
            "Tin tức thị trường",
            "Cài đặt dữ liệu"
        ]
    )

    st.divider()
    st.markdown("### Trạng thái dữ liệu")
    st.warning(
        "Dữ liệu lịch sử / có thể trễ. "
        "Chưa xác nhận dữ liệu thời gian thực."
    )

    st.caption(
        "Ứng dụng phục vụ phân tích và học tập, "
        "không phải khuyến nghị mua bán."
    )

# -------------------- HEADER ----------------------------

st.markdown("# 📈 StockVision Vietnam")
st.caption(
    f"Phân tích thị trường Việt Nam · "
    f"Cập nhật giao diện: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)

# =========================================================
# PAGE 1: MARKET OVERVIEW
# =========================================================

if page == "Tổng quan thị trường":
    st.subheader("Tổng quan thị trường")
    st.info(
        "Lưu ý: Yahoo Finance không đảm bảo dữ liệu chứng khoán "
        "Việt Nam theo thời gian thực. Các chỉ số bên dưới chỉ "
        "được hiển thị khi có dữ liệu hợp lệ; không tự tạo giá chỉ số."
    )

    index_symbols = {
        "VN-Index": "^VNINDEX",
        "VN30": "^VN30",
        "HNX-Index": "^HNXI"
    }

    cols = st.columns(3)

    for i, (name, ticker) in enumerate(index_symbols.items()):
        df, source = get_history(ticker)

        with cols[i]:
            if not df.empty:
                close = float(df["Close"].iloc[-1])
                previous = (
                    float(df["Close"].iloc[-2])
                    if len(df) > 1 else close
                )
                delta = close - previous
                st.metric(
                    name,
                    fmt_price(close),
                    f"{delta:+.2f} ({delta / previous * 100:+.2f}%)"
                    if previous else None
                )
                st.caption("Nguồn: Yahoo Finance · Có thể trễ")
            else:
                st.metric(name, "N/A", "Không có dữ liệu")
                st.caption("Chưa kết nối được nguồn chỉ số")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Top tăng giá")
        table = build_stock_table(POPULAR)
        if not table.empty:
            render_stock_table(
                table.sort_values("% thay đổi", ascending=False).head(5)
            )

    with c2:
        st.subheader("📉 Top giảm giá")
        table = build_stock_table(POPULAR)
        if not table.empty:
            render_stock_table(
                table.sort_values("% thay đổi").head(5)
            )

    st.subheader("🔥 Thanh khoản cao")
    table = build_stock_table(POPULAR)
    if not table.empty:
        render_stock_table(
            table.sort_values("Khối lượng", ascending=False).head(10)
        )

    st.caption(
        "Các bảng cổ phiếu dựa trên phiên lịch sử gần nhất có dữ liệu. "
        "Chúng không đại diện cho bảng giá trực tuyến của sở giao dịch."
    )

# =========================================================
# PAGE 2: STOCK LOOKUP
# =========================================================

elif page == "Tra cứu cổ phiếu":
    st.subheader("Tra cứu cổ phiếu")

    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10
    ).strip().upper()

    period_label = st.selectbox(
        "Khoảng thời gian",
        ["1 tháng", "3 tháng", "6 tháng", "1 năm", "2 năm", "5 năm"],
        index=3
    )

    period_map = {
        "1 tháng": "1mo",
        "3 tháng": "3mo",
        "6 tháng": "6mo",
        "1 năm": "1y",
        "2 năm": "2y",
        "5 năm": "5y"
    }

    if symbol:
        df_raw, source_raw = get_history(symbol, period_map[period_label])

        if df_raw.empty:
            st.warning(
                f"Không lấy được dữ liệu lịch sử cho {symbol}. "
                "Kiểm tra mã hoặc kết nối mạng."
            )
        else:
            df = df_raw
            latest = df.iloc[-1]
            previous = (
                float(df["Close"].iloc[-2])
                if len(df) > 1 else float(latest["Close"])
            )
            current = float(latest["Close"])
            change = current - previous
            pct = change / previous * 100 if previous else 0

            st.markdown(f"### {symbol} — {company_name(symbol)}")
            st.caption(
                f"Sàn dự kiến: {exchange_name(symbol)} · "
                f"Nguồn: Yahoo Finance · "
                f"Phiên dữ liệu cuối: {latest['Date']}"
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Giá đóng cửa", fmt_price(current),
                      f"{change:+.2f} ({pct:+.2f}%)")
            m2.metric("Giá mở cửa", fmt_price(latest["Open"]))
            m3.metric("Cao nhất", fmt_price(latest["High"]))
            m4.metric("Thấp nhất", fmt_price(latest["Low"]))

            st.warning(
                "Giá hiển thị là dữ liệu lịch sử từ nhà cung cấp, "
                "không được xác nhận là giá thời gian thực."
            )

            candle_chart(df, symbol)
            volume_chart(df)
            performance_chart(df)

            st.subheader("Dữ liệu lịch sử")
            st.dataframe(
                df[["Date", "Open", "High", "Low", "Close", "Volume"]]
                .sort_values("Date", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Tải lịch sử giá CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{symbol}_history.csv",
                mime="text/csv"
            )

            if st.button("⭐ Thêm vào danh sách theo dõi"):
                if symbol not in st.session_state.watchlist:
                    st.session_state.watchlist.append(symbol)
                    st.success(f"Đã thêm {symbol} vào danh sách theo dõi.")
                else:
                    st.info("Mã này đã có trong danh sách theo dõi.")

            with st.expander("Chỉ số tài chính doanh nghiệp"):
                fundamentals = get_fundamentals(symbol)
                st.json(fundamentals)

# =========================================================
# PAGE 3: WATCHLIST AND ALERTS
# =========================================================

elif page == "Danh sách theo dõi":
    st.subheader("Danh sách theo dõi")
    st.caption(
        "Cảnh báo được kiểm tra khi ứng dụng chạy và tải dữ liệu; "
        "không phải thông báo đẩy liên tục."
    )

    with st.form("add_watchlist"):
        new_symbol = st.text_input(
            "Thêm mã cổ phiếu", placeholder="Ví dụ: FPT"
        ).strip().upper()
        add_submit = st.form_submit_button("Thêm mã")

    if add_submit and new_symbol:
        if new_symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_symbol)
            st.success(f"Đã thêm {new_symbol}.")
            st.rerun()
        else:
            st.warning("Mã đã có trong danh sách.")

    watch = st.session_state.watchlist.copy()
    table = build_stock_table(watch)
    render_stock_table(table)

    if watch:
        selected_remove = st.selectbox(
            "Chọn mã để xóa",
            ["— Chọn mã —"] + watch
        )
        if st.button("🗑️ Xóa mã khỏi danh sách"):
            if selected_remove != "— Chọn mã —":
                st.session_state.watchlist.remove(selected_remove)
                st.session_state.alerts.pop(selected_remove, None)
                st.rerun()

    st.divider()
    st.subheader("🔔 Cảnh báo giá")

    with st.form("alert_form"):
        alert_symbol = st.selectbox(
            "Mã cổ phiếu",
            watch if watch else ["FPT"]
        )
        target_price = st.number_input(
            "Ngưỡng giá (nghìn đồng)",
            min_value=0.0,
            value=100.0,
            step=0.5
        )
        direction = st.selectbox(
            "Điều kiện",
            ["Giá lớn hơn hoặc bằng ngưỡng",
             "Giá nhỏ hơn hoặc bằng ngưỡng"]
        )
        alert_submit = st.form_submit_button("Lưu cảnh báo")

    if alert_submit:
        st.session_state.alerts[alert_symbol] = {
            "target": target_price,
            "direction": direction
        }
        st.success("Đã lưu ngưỡng cảnh báo.")

    for sym, alert in list(st.session_state.alerts.items()):
        info = get_last_price(sym)

        if not info:
            continue

        triggered = (
            info["price"] >= alert["target"]
            if alert["direction"] == "Giá lớn hơn hoặc bằng ngưỡng"
            else info["price"] <= alert["target"]
        )

        label = "MINH HỌA" if info["demo"] else "Dữ liệu bên ngoài"

        if triggered:
            st.warning(
                f"{sym}: điều kiện ngưỡng đã đạt theo dữ liệu hiện có "
                f"({label}). Giá: {info['price']:.2f}; "
                f"ngưỡng: {alert['target']:.2f} nghìn đồng."
            )
        else:
            st.write(
                f"**{sym}** — Giá: {info['price']:.2f} | "
                f"Ngưỡng: {alert['target']:.2f} | {label}"
            )

        if st.button(f"Xóa cảnh báo {sym}", key=f"del_alert_{sym}"):
            st.session_state.alerts.pop(sym, None)
            st.rerun()

# =========================================================
# PAGE 4: PAPER PORTFOLIO
# =========================================================

elif page == "Danh mục mô phỏng":
    st.subheader("Danh mục đầu tư mô phỏng")
    st.caption(
        "Không sử dụng tiền thật. Giá tham chiếu có thể là giá lịch sử "
        "và chưa tính phí giao dịch, thuế hoặc các điều chỉnh khác."
    )

    with st.expander("Thiết lập vốn mô phỏng"):
        new_capital = st.number_input(
            "Vốn ban đầu (VND)",
            min_value=1_000_000.0,
            max_value=100_000_000_000.0,
            value=float(st.session_state.initial_capital),
            step=1_000_000.0
        )
        if st.button("Cập nhật vốn ban đầu"):
            # Reset only with explicit confirmation
            st.session_state.pending_capital = new_capital
            st.warning(
                "Đổi vốn sẽ đặt lại danh mục và lịch sử giao dịch."
            )

        if "pending_capital" in st.session_state:
            confirm = st.checkbox(
                "Tôi đồng ý xóa danh mục mô phỏng hiện tại",
                key="confirm_reset_portfolio"
            )
            if st.button("Xác nhận đặt lại danh mục"):
                if confirm:
                    st.session_state.initial_capital = (
                        st.session_state.pending_capital
                    )
                    st.session_state.cash = (
                        st.session_state.pending_capital
                    )
                    st.session_state.holdings = {}
                    st.session_state.transactions = []
                    del st.session_state.pending_capital
                    st.success("Đã đặt lại danh mục.")
                    st.rerun()
                else:
                    st.error("Vui lòng xác nhận trước khi đặt lại.")

    st.markdown("### Ghi nhận giao dịch giả lập")

    with st.form("trade_form"):
        trade_symbol = st.text_input(
            "Mã cổ phiếu", value="FPT"
        ).strip().upper()
        trade_side = st.selectbox("Loại giao dịch", ["MUA", "BÁN"])
        trade_qty = st.number_input(
            "Số lượng cổ phiếu",
            min_value=1, max_value=100_000_000, value=100, step=100
        )

        latest_info = get_last_price(trade_symbol) if trade_symbol else None
        default_price = (
            float(latest_info["price"]) if latest_info else 100.0
        )
        trade_price = st.number_input(
            "Giá giao dịch (nghìn đồng/cổ phiếu)",
            min_value=0.01,
            value=max(default_price, 0.01),
            step=0.1
        )

        trade_submit = st.form_submit_button("Xác nhận giao dịch")

    if trade_submit:
        ok, message = execute_trade(
            trade_symbol, trade_side, int(trade_qty), float(trade_price)
        )
        if ok:
            st.success(message)
        else:
            st.error(message)

    holdings = holdings_frame()
    market_value = (
        float(holdings["Giá trị thị trường (đ)"].sum())
        if not holdings.empty else 0.0
    )
    invested_cost = (
        float(holdings["Số lượng"].mul(
            holdings["Giá vốn (nghìn đồng)"]
        ).sum() * 1000)
        if not holdings.empty else 0.0
    )
    total_assets = st.session_state.cash + market_value
    pnl = total_assets - st.session_state.initial_capital
    pnl_pct = (
        pnl / st.session_state.initial_capital * 100
        if st.session_state.initial_capital else 0
    )

    a, b, c, d = st.columns(4)
    a.metric("Tiền mặt", fmt_vnd(st.session_state.cash))
    b.metric("Giá trị cổ phiếu", fmt_vnd(market_value))
    c.metric("Tổng tài sản", fmt_vnd(total_assets))
    d.metric("Lãi/lỗ tổng", fmt_vnd(pnl), f"{pnl_pct:+.2f}%")

    st.subheader("Cổ phiếu đang nắm giữ")
    if holdings.empty:
        st.info("Danh mục chưa có cổ phiếu.")
    else:
        st.dataframe(
            holdings,
            use_container_width=True,
            hide_index=True
        )

        # Display whether the quote data are illustrative.
        if holdings["Minh họa"].any():
            st.warning(
                "Một hoặc nhiều mã đang dùng giá minh họa. "
                "Lãi/lỗ danh mục trong trường hợp này không có giá trị "
                "đại diện cho hiệu suất đầu tư thực tế."
            )

    st.subheader("Lịch sử giao dịch")
    transactions = pd.DataFrame(st.session_state.transactions)

    if transactions.empty:
        st.info("Chưa có giao dịch.")
    else:
        st.dataframe(
            transactions.sort_values("Thời gian", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Xuất lịch sử giao dịch CSV",
            data=transactions.to_csv(index=False).encode("utf-8-sig"),
            file_name="stockvision_transactions.csv",
            mime="text/csv"
        )

# =========================================================
# PAGE 5: ANALYSIS AND EDUCATION
# =========================================================

elif page == "Phân tích & kiến thức":
    st.subheader("Phân tích doanh nghiệp")

    symbol = st.text_input(
        "Mã cần phân tích", value="FPT"
    ).strip().upper()

    if symbol:
        fundamentals = get_fundamentals(symbol)

        st.markdown(f"### {company_name(symbol)} ({symbol})")
        st.caption(f"Nguồn: {fundamentals['Nguồn']}")

        cols = st.columns(4)
        metrics = [
            ("P/E", "P/E"),
            ("P/B", "P/B"),
            ("EPS", "EPS"),
            ("ROE (%)", "ROE (%)")
        ]

        for col, (label, key) in zip(cols, metrics):
            val = fundamentals.get(key)
            col.metric(
                label,
                f"{val:,.2f}" if val is not None else "N/A"
            )

        st.write("**Vốn hóa:**",
                 fmt_vnd(fundamentals["Vốn hóa"])
                 if fundamentals.get("Vốn hóa") is not None
                 else "N/A")

        st.divider()
        st.subheader("Phân tích dữ liệu giá")

        df, source, demo = get_stock(symbol)
        if not df.empty:
            lookback = st.slider(
                "Số phiên để phân tích",
                min_value=20,
                max_value=min(250, len(df)),
                value=min(60, len(df))
            )
            sub = df.tail(lookback).copy()

            daily_returns = sub["Close"].pct_change().dropna()
            total_return = (
                sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1
            ) * 100
            volatility = (
                daily_returns.std() * np.sqrt(252) * 100
                if len(daily_returns) > 1 else 0
            )
            ma20 = sub["Close"].rolling(20).mean().iloc[-1]
            ma50 = sub["Close"].rolling(50).mean().iloc[-1]

            x1, x2, x3 = st.columns(3)
            x1.metric("Biến động giá kỳ (%)", f"{total_return:+.2f}%")
            x2.metric("Biến động năm hóa (%)", f"{volatility:.2f}%")
            x3.metric("Trung bình 20 phiên", fmt_price(ma20))

            st.write(
                "**Trung bình 50 phiên:**",
                fmt_price(ma50) if pd.notna(ma50) else "Chưa đủ dữ liệu"
            )

            if demo:
                st.warning(
                    "Kết quả đang dựa trên dữ liệu MINH HỌA, "
                    "không dùng để đánh giá cổ phiếu thật."
                )
            performance_chart(sub)

    st.divider()
    st.subheader("📚 Kiến thức cho người mới")

    with st.expander("P/E — Giá trên lợi nhuận"):
        st.write(
            "P/E = Giá cổ phiếu / EPS. Chỉ số này cho biết thị trường "
            "đang trả bao nhiêu đồng cho mỗi đồng lợi nhuận trên cổ phiếu. "
            "P/E cao không tự động có nghĩa là cổ phiếu đắt."
        )

    with st.expander("P/B — Giá trên giá trị sổ sách"):
        st.write(
            "P/B = Giá cổ phiếu / Giá trị sổ sách trên mỗi cổ phiếu. "
            "Cần so sánh với doanh nghiệp cùng ngành và xem xét chất lượng "
            "tài sản, nợ phải trả."
        )

    with st.expander("EPS — Lợi nhuận trên mỗi cổ phiếu"):
        st.write(
            "EPS thể hiện phần lợi nhuận quy thuộc cho mỗi cổ phiếu "
            "theo cách tính của doanh nghiệp. EPS tăng không phải lúc nào "
            "cũng đồng nghĩa hoạt động kinh doanh tốt hơn."
        )

    with st.expander("ROE — Tỷ suất sinh lời trên vốn chủ sở hữu"):
        st.write(
            "ROE = Lợi nhuận sau thuế / Vốn chủ sở hữu bình quân "
            "(tùy cách tính). ROE cần được đánh giá cùng mức nợ và "
            "đặc thù ngành."
        )

    with st.expander("Rủi ro khi dự báo giá"):
        st.write(
            "Giá cổ phiếu chịu tác động bởi lợi nhuận doanh nghiệp, "
            "lãi suất, dòng tiền, chính sách, tâm lý thị trường và nhiều "
            "yếu tố khó dự đoán. Phân tích kỹ thuật hay dữ liệu lịch sử "
            "không bảo đảm dự báo chính xác tương lai."
        )

# =========================================================
# PAGE 6: NEWS
# =========================================================

elif page == "Tin tức thị trường":
    st.subheader("📰 Tin tức thị trường")
    query = st.text_input(
        "Tìm tin tức",
        value="chứng khoán Việt Nam"
    )

    if st.button("Tải tin tức"):
        st.cache_data.clear()

    articles = get_news(query)

    if not articles:
        st.warning(
            "Chưa lấy được tin. Kiểm tra kết nối Internet "
            "hoặc thử lại sau."
        )
    else:
        for article in articles:
            title = html.escape(article["title"])
            link = html.escape(article["link"], quote=True)
            published = html.escape(article["published"])
            source = html.escape(str(article["source"]))

            st.markdown(
                f"""
                <div class="market-card" style="margin-bottom:12px;">
                    <a href="{link}" target="_blank"
                       style="font-size:17px;font-weight:600;">
                        {title}
                    </a>
                    <p class="small-note">
                        Nguồn: {source} · {published}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "Tin được lấy qua Google News RSS. Hãy mở bài gốc để "
            "kiểm tra nội dung, thời điểm xuất bản và độ tin cậy."
        )

# =========================================================
# PAGE 7: DATA SETTINGS
# =========================================================

elif page == "Cài đặt dữ liệu":
    st.subheader("⚙️ Cài đặt dữ liệu")

    st.markdown("### Nguồn dữ liệu hiện tại")
    st.write("- Giá lịch sử: Yahoo Finance thông qua `yfinance`.")
    st.write("- Tin tức: Google News RSS.")
    st.write("- Dữ liệu chỉ số: phụ thuộc khả năng cung cấp của Yahoo Finance.")
    st.write("- Dữ liệu minh họa: chỉ sử dụng tại nơi có ghi rõ.")

    st.warning(
        "Phiên bản hiện tại CHƯA có kết nối API chứng khoán Việt Nam "
        "được xác nhận là thời gian thực. Không dùng dữ liệu này để "
        "đặt lệnh giao dịch thực tế."
    )

    st.markdown("### Muốn kết nối dữ liệu thời gian thực?")
    st.write(
        "Bạn cần đăng ký nhà cung cấp dữ liệu có quyền cung cấp "
        "bảng giá trực tuyến cho thị trường Việt Nam. Kiểm tra các "
        "điều khoản về độ trễ, quyền sử dụng dữ liệu, giới hạn API "
        "và quyền hiển thị dữ liệu cho người dùng."
    )

    st.markdown("### Quản lý cache")
    if st.button("Làm mới cache dữ liệu"):
        st.cache_data.clear()
        st.success(
            "Đã yêu cầu xóa cache. Tải lại trang để lấy dữ liệu mới."
        )

    st.markdown("### Thông tin hệ thống")
    st.write(f"Python: {__import__('sys').version.split()[0]}")
    st.write(f"Streamlit: {st.__version__}")
    st.write(f"Thời gian máy chủ: {datetime.now().isoformat(timespec='seconds')}")

# -------------------- FOOTER ----------------------------

st.divider()
st.caption(
    "StockVision Vietnam · Công cụ phân tích và mô phỏng học tập. "
    "Không phải tư vấn đầu tư. Không cam kết lợi nhuận. "
    "Giá và chỉ số có thể trễ, thiếu hoặc không chính xác."
)
