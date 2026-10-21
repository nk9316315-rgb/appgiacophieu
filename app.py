
# -*- coding: utf-8 -*-
# VietStock Terminal - EOD (End of Day)
# Chay: streamlit run app.py

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# =========================================================
# 1. CAU HINH
# =========================================================
st.set_page_config(
    page_title="VietStock Terminal",
    page_icon="📈",
    layout="wide",
)

DB_FILE = "vietstock.db"

POPULAR = [
    "FPT", "SSI", "VNM", "VCB", "HPG",
    "MWG", "ACB", "TCB", "MBB", "VIC"
]

# Chi dung cac ma Yahoo Finance da co the xac minh.
# Mã chỉ số có thể không được nguồn hỗ trợ.
SYMBOLS = {
    "VN-Index": "^VNINDEX",
    "VN30": "^VN30",
    "HNX-Index": "^HNX",
}

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #e8edf7;
}
[data-testid="stSidebar"] {
    background: #11182b;
}
[data-testid="stMetric"] {
    background: #121b30;
    border: 1px solid #283650;
    border-radius: 12px;
    padding: 12px;
}
h1, h2, h3 { letter-spacing: -0.5px; }
.small-note { color: #9eacc5; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. DATABASE - LUU DU LIEU LAU DAI
# =========================================================
def db():
    con = sqlite3.connect(DB_FILE, timeout=15)
    con.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            target REAL NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_time TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            note TEXT DEFAULT ''
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    con.commit()
    return con


def read_sql(query, params=()):
    with db() as con:
        return pd.read_sql_query(query, con, params=params)


def add_watch(symbol):
    with db() as con:
        con.execute(
            "INSERT OR IGNORE INTO watchlist(symbol) VALUES (?)",
            (symbol.upper(),)
        )


def remove_watch(symbol):
    with db() as con:
        con.execute("DELETE FROM watchlist WHERE symbol=?", (symbol,))


def get_setting(key, default):
    with db() as con:
        row = con.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
    return row[0] if row else str(default)


def set_setting(key, value):
    with db() as con:
        con.execute("""
            INSERT INTO settings(key,value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, str(value)))


# =========================================================
# 3. DU LIEU CUOI NGAY
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_history(symbol, period="1y"):
    """
    Chi lay nen ngay (1d).
    Nguon Yahoo Finance co the khong ho tro mot so ma VN.
    Khong tao gia gia khi khong co du lieu.
    """
    if yf is None:
        raise RuntimeError(
            "Chua cai yfinance. Chay: pip install yfinance"
        )

    ticker = symbol.upper()
    if not ticker.startswith("^") and not ticker.endswith(".VN"):
        ticker += ".VN"

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"Nguon du lieu khong tra lich su cho {ticker}. "
            "Thu ma khac hoac ket noi nha cung cap API khac."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"Thieu cot du lieu: {missing}")

    if "Volume" not in data.columns:
        data["Volume"] = 0

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"]).sort_index()

    if data.empty:
        raise RuntimeError("Khong co du lieu gia hop le.")

    return data


def snapshot(symbol):
    try:
        data = load_history(symbol, "3mo")
        if len(data) < 1:
            return None

        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else last
        close = float(last["Close"])
        old = float(prev["Close"])

        return {
            "Mã": symbol,
            "Giá đóng cửa": close,
            "Thay đổi": close - old,
            "% thay đổi": (close / old - 1) * 100 if old else 0,
            "Khối lượng": float(last["Volume"]),
            "Ngày dữ liệu": data.index[-1].strftime("%d/%m/%Y"),
        }
    except Exception:
        return None


def indicators(data):
    df = data.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def forecast(data, days):
    """
    Duong xu huong log-gia don gian.
    Day la mo hinh co so, khong phai loi hua ve gia tuong lai.
    """
    prices = data["Close"].dropna().tail(120).astype(float)

    if len(prices) < 20:
        raise ValueError("Can toi thieu 20 phien de du bao.")

    x = np.arange(len(prices), dtype=float)
    y = np.log(prices.values)
    slope, intercept = np.polyfit(x, y, 1)

    fitted = intercept + slope * x
    sigma = max(float(np.std(y - fitted, ddof=1)), 0.001)

    future_x = np.arange(len(prices), len(prices) + days)
    predicted = np.exp(intercept + slope * future_x)
    uncertainty = sigma * np.sqrt(
        1 + np.arange(1, days + 1) / len(prices)
    )

    # Ngay lam viec tham khao, khong xu ly ngay nghi le VN.
    dates = pd.bdate_range(
        pd.Timestamp(prices.index[-1]) + pd.Timedelta(days=1),
        periods=days,
    )

    return pd.DataFrame({
        "Ngày tham khảo": dates,
        "Giá dự báo": predicted,
        "Khoảng thấp": predicted * np.exp(-1.96 * uncertainty),
        "Khoảng cao": predicted * np.exp(1.96 * uncertainty),
    })


def money(value):
    return f"{value:,.0f}" if pd.notna(value) else "—"


# =========================================================
# 4. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("📈 VietStock")
    st.caption("EOD MARKET RESEARCH")
    page = st.radio(
        "ĐIỀU HƯỚNG",
        [
            "Tổng quan thị trường",
            "Tra cứu cổ phiếu",
            "Danh sách theo dõi",
            "Danh mục giả lập",
            "Phân tích & dự báo",
            "Dữ liệu & cài đặt",
        ],
    )
    st.divider()
    st.caption("Chế độ: dữ liệu lịch sử ngày")
    st.caption("Không phải bảng giá realtime")


st.title("VietStock Terminal")
st.caption(
    "Nền tảng theo dõi, phân tích và dự báo cổ phiếu Việt Nam"
)
st.info(
    "Dữ liệu cuối ngày (EOD). Nguồn dự phòng có thể không hỗ trợ "
    "mọi mã chứng khoán Việt Nam. Luôn kiểm tra ngày dữ liệu cuối."
)


# =========================================================
# 5. TONG QUAN THI TRUONG
# =========================================================
if page == "Tổng quan thị trường":
    st.subheader("Chỉ số thị trường")

    index_cols = st.columns(3)
    for col, (name, ticker) in zip(index_cols, SYMBOLS.items()):
        with col:
            info = snapshot(ticker)
            if info:
                col.metric(
                    name,
                    money(info["Giá đóng cửa"]),
                    f'{info["% thay đổi"]:+.2f}%'
                )
                col.caption(f'Phiên: {info["Ngày dữ liệu"]}')
            else:
                col.metric(name, "Chưa có dữ liệu")
                col.caption("Nguồn hiện tại có thể không hỗ trợ mã này.")

    st.divider()
    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Biểu đồ thị trường")
        index_name = st.selectbox("Chỉ số", list(SYMBOLS.keys()))
        period = st.selectbox(
            "Khoảng thời gian",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
        )
        try:
            df = load_history(SYMBOLS[index_name], period)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Close"],
                mode="lines", name=index_name
            ))
            fig.update_layout(
                template="plotly_dark",
                height=370,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Ngày",
                yaxis_title="Giá/điểm",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f'Dữ liệu cuối: {df.index[-1].strftime("%d/%m/%Y")}'
            )
        except Exception as exc:
            st.warning(f"Không tải được chỉ số: {exc}")

    with right:
        st.subheader("Bảng giá khảo sát")
        symbols_text = st.text_input(
            "Nhập mã, cách nhau bằng dấu phẩy",
            "FPT,SSI,VNM,VCB,HPG",
        )
        symbols = list(dict.fromkeys(
            x.strip().upper()
            for x in symbols_text.split(",")
            if x.strip()
        ))[:15]

        if st.button("Tải bảng giá"):
            st.session_state["market_symbols"] = symbols

        symbols = st.session_state.get("market_symbols", symbols)
        with st.spinner("Đang tải dữ liệu EOD..."):
            rows = [
                item for item in (snapshot(s) for s in symbols)
                if item is not None
            ]

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Không có dữ liệu từ nguồn hiện tại.")

    st.divider()
    st.subheader("Top tăng, giảm và khối lượng")
    with st.spinner("Đang tổng hợp..."):
        rows = [
            item for item in (snapshot(s) for s in POPULAR)
            if item is not None
        ]

    if rows:
        market = pd.DataFrame(rows)
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Tăng mạnh**")
            st.dataframe(
                market.sort_values("% thay đổi", ascending=False)
                .head(5)[["Mã", "Giá đóng cửa", "% thay đổi"]],
                use_container_width=True, hide_index=True
            )
        with b:
            st.markdown("**Giảm mạnh**")
            st.dataframe(
                market.sort_values("% thay đổi")
                .head(5)[["Mã", "Giá đóng cửa", "% thay đổi"]],
                use_container_width=True, hide_index=True
            )
        with c:
            st.markdown("**Khối lượng cao**")
            st.dataframe(
                market.sort_values("Khối lượng", ascending=False)
                .head(5)[["Mã", "Giá đóng cửa", "Khối lượng"]],
                use_container_width=True, hide_index=True
            )


# =========================================================
# 6. TRA CUU CO PHIEU
# =========================================================
elif page == "Tra cứu cổ phiếu":
    st.subheader("Tra cứu và biểu đồ lịch sử")

    col1, col2, col3 = st.columns(3)
    symbol = col1.text_input("Mã cổ phiếu", "FPT").strip().upper()
    period = col2.selectbox(
        "Khoảng thời gian",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )
    chart_type = col3.selectbox("Biểu đồ", ["Nến", "Đường"])

    if symbol:
        try:
            data = load_history(symbol, period)
            df = indicators(data)
            last = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else last
            change = (last["Close"] / previous["Close"] - 1) * 100

            metrics = st.columns(5)
            metrics[0].metric(
                "Đóng cửa", money(last["Close"]), f"{change:+.2f}%"
            )
            metrics[1].metric("Mở cửa", money(last["Open"]))
            metrics[2].metric("Cao nhất", money(last["High"]))
            metrics[3].metric("Thấp nhất", money(last["Low"]))
            metrics[4].metric("Khối lượng", money(last["Volume"]))

            st.caption(
                f'Ngày dữ liệu cuối: {df.index[-1].strftime("%d/%m/%Y")}. '
                "Giá đóng cửa của phiên cuối có thể thay đổi theo điều chỉnh "
                "của nhà cung cấp."
            )

            if st.button(f"☆ Theo dõi {symbol}"):
                add_watch(symbol)
                st.success(f"Đã thêm {symbol} vào danh sách theo dõi.")

            fig = go.Figure()
            if chart_type == "Nến":
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"],
                    name=symbol,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["Close"],
                    mode="lines", name="Giá đóng cửa",
                ))

            fig.add_trace(go.Scatter(
                x=df.index, y=df["MA20"], name="MA20"
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["MA50"], name="MA50"
            ))
            fig.update_layout(
                template="plotly_dark",
                height=480,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            volfig = go.Figure(go.Bar(
                x=df.index, y=df["Volume"], name="Khối lượng"
            ))
            volfig.update_layout(
                template="plotly_dark", height=220
            )
            st.plotly_chart(volfig, use_container_width=True)

            a, b, c = st.columns(3)
            a.metric(
                "RSI 14",
                f'{last["RSI"]:.2f}' if pd.notna(last["RSI"]) else "N/A"
            )
            b.metric(
                "MACD",
                f'{last["MACD"]:.2f}' if pd.notna(last["MACD"]) else "N/A"
            )
            c.metric("MA20", money(last["MA20"]))

            with st.expander("Xem lịch sử giá"):
                hist = df.reset_index().rename(
                    columns={df.index.name or "index": "Ngày"}
                )
                st.dataframe(
                    hist.sort_values("Ngày", ascending=False),
                    use_container_width=True, hide_index=True
                )
                st.download_button(
                    "Tải lịch sử CSV",
                    hist.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{symbol}_history.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(str(exc))


# =========================================================
# 7. DANH SACH THEO DOI VA CANH BAO
# =========================================================
elif page == "Danh sách theo dõi":
    st.subheader("Danh sách theo dõi")
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("Thêm mã cổ phiếu", "FPT").strip().upper()
        if st.button("Thêm mã"):
            if symbol:
                add_watch(symbol)
                st.success(f"Đã thêm {symbol}.")
                st.rerun()

        watch = read_sql("SELECT symbol FROM watchlist")
        if watch.empty:
            st.info("Chưa có mã nào được theo dõi.")
        else:
            rows = [
                item for item in (
                    snapshot(s) for s in watch["symbol"].tolist()
                ) if item is not None
            ]
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

            remove_symbol = st.selectbox(
                "Chọn mã cần xóa", watch["symbol"].tolist()
            )
            if st.button("Xóa mã đã chọn"):
                remove_watch(remove_symbol)
                st.rerun()

    with col2:
        st.subheader("Cảnh báo giá")
        watch_symbols = read_sql(
            "SELECT symbol FROM watchlist"
        )["symbol"].tolist()
        alert_symbol = st.selectbox(
            "Mã cảnh báo",
            watch_symbols if watch_symbols else POPULAR,
        )
        direction = st.selectbox(
            "Điều kiện",
            ["Giá >= ngưỡng", "Giá <= ngưỡng"],
        )
        target = st.number_input(
            "Ngưỡng giá (đồng)", min_value=0.0, value=100000.0
        )

        if st.button("Lưu cảnh báo"):
            with db() as con:
                con.execute(
                    "INSERT INTO alerts(symbol,direction,target) VALUES(?,?,?)",
                    (
                        alert_symbol,
                        "above" if ">=" in direction else "below",
                        target,
                    ),
                )
            st.success("Đã lưu cảnh báo.")

        alerts = read_sql("SELECT * FROM alerts ORDER BY id DESC")
        if not alerts.empty:
            st.dataframe(
                alerts, use_container_width=True, hide_index=True
            )
            for _, alert in alerts.iterrows():
                info = snapshot(alert["symbol"])
                if info:
                    price = info["Giá đóng cửa"]
                    hit = (
                        price >= alert["target"]
                        if alert["direction"] == "above"
                        else price <= alert["target"]
                    )
                    if hit:
                        st.warning(
                            f'{alert["symbol"]}: giá cuối ngày '
                            f'{money(price)} đã chạm ngưỡng '
                            f'{money(alert["target"])}.'
                        )

            alert_id = st.number_input(
                "ID cảnh báo cần xóa", min_value=1, step=1
            )
            if st.button("Xóa cảnh báo"):
                with db() as con:
                    con.execute(
                        "DELETE FROM alerts WHERE id=?", (int(alert_id),)
                    )
                st.rerun()

        st.caption(
            "Cảnh báo được kiểm tra khi mở/tải lại ứng dụng. "
            "Không phải thông báo đẩy liên tục 24/7."
        )


# =========================================================
# 8. DANH MUC DAU TU GIA LAP
# =========================================================
elif page == "Danh mục giả lập":
    st.subheader("Danh mục đầu tư giả lập")

    initial = float(get_setting("capital", 100_000_000))
    with st.expander("Thiết lập vốn ban đầu"):
        new_capital = st.number_input(
            "Vốn ban đầu (đồng)",
            min_value=0.0, value=initial, step=1_000_000.0,
        )
        if st.button("Lưu vốn"):
            set_setting("capital", new_capital)
            st.rerun()

    with st.form("trade_form"):
        symbol = st.text_input("Mã cổ phiếu", "FPT").strip().upper()
        side = st.selectbox("Giao dịch", ["MUA", "BÁN"])
        qty = st.number_input(
            "Số lượng cổ phiếu", min_value=1, value=100, step=100
        )
        price = st.number_input(
            "Giá khớp giả lập (đồng)",
            min_value=1.0, value=100000.0, step=100.0,
        )
        fee = st.number_input("Phí giao dịch (đồng)", min_value=0.0)
        note = st.text_input("Ghi chú")
        submit = st.form_submit_button("Ghi giao dịch")

    trades = read_sql("SELECT * FROM trades ORDER BY id")
    capital = float(get_setting("capital", 100_000_000))

    # Tinh so du va vi the theo lich su giao dich
    cash = capital
    positions = {}
    realized = 0.0

    for _, tr in trades.iterrows():
        sym = tr["symbol"]
        q = int(tr["quantity"])
        p = float(tr["price"])
        f = float(tr["fee"])
        pos = positions.setdefault(sym, {"qty": 0, "cost": 0.0})

        if tr["side"] == "MUA":
            cash -= q * p + f
            pos["cost"] += q * p + f
            pos["qty"] += q
        else:
            if q > pos["qty"]:
                continue
            avg = pos["cost"] / pos["qty"] if pos["qty"] else 0
            cash += q * p - f
            realized += q * p - f - avg * q
            pos["cost"] -= avg * q
            pos["qty"] -= q

    if submit:
        if not symbol or price <= 0:
            st.error("Nhập mã và giá hợp lệ.")
        elif side == "MUA" and qty * price + fee > cash:
            st.error("Không đủ tiền mặt giả lập.")
        elif side == "BÁN" and qty > positions.get(
            symbol, {"qty": 0}
        )["qty"]:
            st.error("Số lượng bán lớn hơn số lượng đang nắm giữ.")
        else:
            with db() as con:
                con.execute("""
                    INSERT INTO trades(
                        trade_time,symbol,side,quantity,price,fee,note
                    ) VALUES(?,?,?,?,?,?,?)
                """, (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol, side, int(qty), float(price), float(fee), note,
                ))
            st.success("Đã ghi giao dịch giả lập.")
            st.rerun()

    rows = []
    market_value = 0.0
    unrealized = 0.0

    for sym, pos in positions.items():
        if pos["qty"] <= 0:
            continue

        snap = snapshot(sym)
        market_price = (
            snap["Giá đóng cửa"]
            if snap else pos["cost"] / pos["qty"]
        )
        avg = pos["cost"] / pos["qty"]
        value = pos["qty"] * market_price
        pnl = value - pos["cost"]
        market_value += value
        unrealized += pnl

        rows.append({
            "Mã": sym,
            "Số lượng": pos["qty"],
            "Giá vốn TB": avg,
            "Giá cuối ngày": market_price,
            "Giá trị": value,
            "Lãi/lỗ chưa thực hiện": pnl,
        })

    a, b, c, d = st.columns(4)
    a.metric("Tiền mặt", f"{cash:,.0f} đ")
    b.metric("Giá trị cổ phiếu", f"{market_value:,.0f} đ")
    c.metric("Tổng tài sản", f"{cash + market_value:,.0f} đ")
    d.metric("Lãi/lỗ đã thực hiện", f"{realized:,.0f} đ")

    st.metric("Lãi/lỗ chưa thực hiện", f"{unrealized:,.0f} đ")

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Chưa có cổ phiếu trong danh mục.")

    st.subheader("Lịch sử giao dịch")
    trades = read_sql("SELECT * FROM trades ORDER BY id DESC")
    if not trades.empty:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Tải lịch sử giao dịch CSV",
            trades.to_csv(index=False).encode("utf-8-sig"),
            file_name="giao_dich_gia_lap.csv",
            mime="text/csv",
        )


# =========================================================
# 9. PHAN TICH VA DU BAO
# =========================================================
elif page == "Phân tích & dự báo":
    st.subheader("Phân tích kỹ thuật và dự báo tham khảo")
    symbol = st.text_input("Mã cổ phiếu", "FPT").strip().upper()
    period = st.selectbox(
        "Lịch sử dùng để phân tích", ["6mo", "1y", "2y", "5y"]
    )
    horizon = st.slider("Số phiên dự báo", 1, 20, 5)

    if st.button("Chạy phân tích", type="primary"):
        st.session_state["analyze"] = True

    if st.session_state.get("analyze") and symbol:
        try:
            data = load_history(symbol, period)
            df = indicators(data)
            prediction = forecast(data, horizon)
            last = df.iloc[-1]

            a, b, c = st.columns(3)
            a.metric("Giá cuối ngày", money(last["Close"]))
            b.metric(
                "RSI (14)",
                f'{last["RSI"]:.2f}' if pd.notna(last["RSI"]) else "N/A"
            )
            c.metric(
                "MACD",
                f'{last["MACD"]:.2f}' if pd.notna(last["MACD"]) else "N/A"
            )

            fig = go.Figure()
            recent = df.tail(120)
            fig.add_trace(go.Scatter(
                x=recent.index, y=recent["Close"],
                mode="lines", name="Giá lịch sử"
            ))
            fig.add_trace(go.Scatter(
                x=[recent.index[-1]] + list(prediction["Ngày tham khảo"]),
                y=[float(recent["Close"].iloc[-1])] +
                  list(prediction["Giá dự báo"]),
                mode="lines+markers", name="Dự báo cơ sở"
            ))
            fig.add_trace(go.Scatter(
                x=list(prediction["Ngày tham khảo"]) +
                  list(prediction["Ngày tham khảo"][::-1]),
                y=list(prediction["Khoảng cao"]) +
                  list(prediction["Khoảng thấp"][::-1]),
                fill="toself",
                mode="lines",
                name="Khoảng bất định tham khảo",
                line=dict(width=0),
                opacity=0.2,
            ))
            fig.update_layout(
                template="plotly_dark",
                height=450,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                prediction, use_container_width=True, hide_index=True
            )
            st.download_button(
                "Tải dự báo CSV",
                prediction.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{symbol}_du_bao.csv",
                mime="text/csv",
            )

            st.warning(
                "Dự báo sử dụng hồi quy xu hướng đơn giản trên log giá, "
                "chưa được kiểm định đủ để sử dụng trong giao dịch thực tế. "
                "Khoảng bất định không bảo đảm bao phủ giá tương lai."
            )
        except Exception as exc:
            st.error(f"Không thể phân tích: {exc}")

    st.divider()
    st.subheader("Thuật ngữ cho người mới")
    with st.expander("P/E, P/B, EPS và ROE"):
        st.write("**P/E:** Giá cổ phiếu chia cho lợi nhuận trên mỗi cổ phiếu.")
        st.write("**P/B:** Giá cổ phiếu chia cho giá trị sổ sách trên mỗi cổ phiếu.")
        st.write("**EPS:** Lợi nhuận phân bổ cho mỗi cổ phiếu phổ thông.")
        st.write("**ROE:** Lợi nhuận sau thuế so với vốn chủ sở hữu.")
        st.caption(
            "Bản này chưa tích hợp API báo cáo tài chính nên không tự "
            "tạo các chỉ số cơ bản doanh nghiệp."
        )


# =========================================================
# 10. CAI DAT VA TRANG THAI DU LIEU
# =========================================================
elif page == "Dữ liệu & cài đặt":
    st.subheader("Nguồn dữ liệu và lưu trữ")

    st.markdown("""
    **Chế độ dữ liệu:** EOD - nến ngày.

    - Nguồn dự phòng: Yahoo Finance qua `yfinance`, nếu mã được hỗ trợ.
    - Không đảm bảo Yahoo Finance có đủ mã cổ phiếu hoặc chỉ số Việt Nam.
    - Chưa tích hợp xác thực SSI FastConnect trong phiên bản này.
    - Chưa có API báo cáo tài chính và tin tức doanh nghiệp.
    - SQLite lưu danh sách theo dõi, cảnh báo và giao dịch giả lập.
    """)

    st.markdown("### Kiểm tra nguồn dữ liệu")
    test_symbol = st.text_input("Mã cần kiểm tra", "FPT").strip().upper()
    if st.button("Kiểm tra dữ liệu"):
        try:
            test = load_history(test_symbol, "3mo")
            st.success(
                f"Nhận được {len(test)} phiên. "
                f"Phiên cuối: {test.index[-1].strftime('%d/%m/%Y')}."
            )
            st.dataframe(test.tail(10), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    st.markdown("### Lưu ý khi triển khai")
    st.write(
        "Khi đưa lên môi trường cloud, SQLite có thể không bền vững "
        "nếu ổ đĩa bị làm mới. Nên dùng PostgreSQL cho dữ liệu dài hạn."
    )
    st.write(
        "Muốn có dữ liệu EOD đáng tin cậy cho thị trường Việt Nam, "
        "hãy kết nối API nhà cung cấp có quyền sử dụng phù hợp, "
        "ví dụ SSI FastConnect sau khi được cấp credentials."
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "VietStock Terminal · Công cụ nghiên cứu/học tập, không phải "
    "khuyến nghị đầu tư. Kiểm tra nguồn, độ trễ và điều khoản dữ liệu."
)

