"""Stock universe: live S&P 500 / NASDAQ-100 / Dow 30 constituents with a
built-in large-cap fallback for when the live fetch fails."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

INDEX_SOURCES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "Dow 30": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
}
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

ETFS = {
    "SPY": ("S&P 500 ETF", "Broad Market"),
    "QQQ": ("Nasdaq 100 ETF", "Broad Market"),
    "IWM": ("Russell 2000 ETF", "Small Caps"),
    "SCHD": ("US Dividend ETF", "Dividend"),
    "AGG": ("US Bond ETF", "Bonds"),
    "TLT": ("US 20Y Treasury ETF", "Bonds"),
    "IEF": ("US 7-10Y Treasury ETF", "Bonds"),
    "GLD": ("Gold ETF", "Gold"),
}

# Used only if every live Wikipedia fetch fails (no internet, layout change).
FALLBACK = {
    "AAPL": ("Apple", "Information Technology"),
    "MSFT": ("Microsoft", "Information Technology"),
    "NVDA": ("Nvidia", "Information Technology"),
    "AVGO": ("Broadcom", "Information Technology"),
    "AMD": ("AMD", "Information Technology"),
    "CRM": ("Salesforce", "Information Technology"),
    "ORCL": ("Oracle", "Information Technology"),
    "ADBE": ("Adobe", "Information Technology"),
    "CSCO": ("Cisco", "Information Technology"),
    "INTC": ("Intel", "Information Technology"),
    "QCOM": ("Qualcomm", "Information Technology"),
    "TXN": ("Texas Instruments", "Information Technology"),
    "IBM": ("IBM", "Information Technology"),
    "NOW": ("ServiceNow", "Information Technology"),
    "GOOGL": ("Alphabet", "Communication Services"),
    "META": ("Meta Platforms", "Communication Services"),
    "NFLX": ("Netflix", "Communication Services"),
    "DIS": ("Walt Disney", "Communication Services"),
    "TMUS": ("T-Mobile US", "Communication Services"),
    "VZ": ("Verizon", "Communication Services"),
    "AMZN": ("Amazon", "Consumer Discretionary"),
    "TSLA": ("Tesla", "Consumer Discretionary"),
    "HD": ("Home Depot", "Consumer Discretionary"),
    "MCD": ("McDonald's", "Consumer Discretionary"),
    "NKE": ("Nike", "Consumer Discretionary"),
    "SBUX": ("Starbucks", "Consumer Discretionary"),
    "LOW": ("Lowe's", "Consumer Discretionary"),
    "BKNG": ("Booking Holdings", "Consumer Discretionary"),
    "WMT": ("Walmart", "Consumer Staples"),
    "COST": ("Costco", "Consumer Staples"),
    "PG": ("Procter & Gamble", "Consumer Staples"),
    "KO": ("Coca-Cola", "Consumer Staples"),
    "PEP": ("PepsiCo", "Consumer Staples"),
    "PM": ("Philip Morris", "Consumer Staples"),
    "MDLZ": ("Mondelez", "Consumer Staples"),
    "LLY": ("Eli Lilly", "Health Care"),
    "UNH": ("UnitedHealth", "Health Care"),
    "JNJ": ("Johnson & Johnson", "Health Care"),
    "ABBV": ("AbbVie", "Health Care"),
    "MRK": ("Merck", "Health Care"),
    "PFE": ("Pfizer", "Health Care"),
    "TMO": ("Thermo Fisher", "Health Care"),
    "ABT": ("Abbott Labs", "Health Care"),
    "AMGN": ("Amgen", "Health Care"),
    "JPM": ("JPMorgan Chase", "Financials"),
    "V": ("Visa", "Financials"),
    "MA": ("Mastercard", "Financials"),
    "BRK-B": ("Berkshire Hathaway", "Financials"),
    "BAC": ("Bank of America", "Financials"),
    "WFC": ("Wells Fargo", "Financials"),
    "GS": ("Goldman Sachs", "Financials"),
    "MS": ("Morgan Stanley", "Financials"),
    "AXP": ("American Express", "Financials"),
    "BLK": ("BlackRock", "Financials"),
    "XOM": ("Exxon Mobil", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "COP": ("ConocoPhillips", "Energy"),
    "SLB": ("Schlumberger", "Energy"),
    "CAT": ("Caterpillar", "Industrials"),
    "HON": ("Honeywell", "Industrials"),
    "GE": ("GE Aerospace", "Industrials"),
    "UNP": ("Union Pacific", "Industrials"),
    "BA": ("Boeing", "Industrials"),
    "DE": ("Deere", "Industrials"),
    "RTX": ("RTX Corp", "Industrials"),
    "LMT": ("Lockheed Martin", "Industrials"),
    "UPS": ("UPS", "Industrials"),
    "LIN": ("Linde", "Materials"),
    "SHW": ("Sherwin-Williams", "Materials"),
    "FCX": ("Freeport-McMoRan", "Materials"),
    "NEE": ("NextEra Energy", "Utilities"),
    "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Company", "Utilities"),
    "PLD": ("Prologis", "Real Estate"),
    "AMT": ("American Tower", "Real Estate"),
}


def _fetch_index(name: str) -> pd.DataFrame:
    """Live constituents of one index from Wikipedia. Column names differ per
    page, so detect ticker/name/sector columns flexibly."""
    resp = requests.get(INDEX_SOURCES[name], timeout=15, headers=UA_HEADERS)
    resp.raise_for_status()
    for tbl in pd.read_html(StringIO(resp.text)):
        cols = {str(c).strip().lower(): c for c in tbl.columns}
        sym = next((cols[k] for k in ("symbol", "ticker") if k in cols), None)
        nam = next((cols[k] for k in ("security", "company") if k in cols), None)
        sec = next((cols[k] for k in ("gics sector", "sector", "industry")
                    if k in cols), None)
        if sym is None or nam is None:
            continue
        out = pd.DataFrame({
            "Ticker": tbl[sym].astype(str).str.replace(".", "-", regex=False).str.strip(),
            "Name": tbl[nam].astype(str).str.strip(),
            "Sector": tbl[sec].astype(str).str.strip() if sec is not None else "—",
        }).dropna()
        out = out[out["Ticker"].str.match(r"^[A-Z0-9\-]{1,6}$")]
        if len(out) >= 25:  # skip non-constituent tables on the same page
            out["Index"] = name
            return out
    raise ValueError(f"no constituents table found for {name}")


def stocks_table(indices: tuple[str, ...] = ("S&P 500",)) -> pd.DataFrame:
    """Union of the selected indices' members (deduplicated). Falls back to a
    built-in large-cap list if every fetch fails."""
    frames = []
    for ix in indices:
        try:
            frames.append(_fetch_index(ix))
        except Exception:
            continue
    if not frames:
        fb = pd.DataFrame([(t, n, s) for t, (n, s) in FALLBACK.items()],
                          columns=["Ticker", "Name", "Sector"])
        fb["Index"] = "Fallback"
        frames = [fb]
    merged = (pd.concat(frames)
              .groupby("Ticker")
              .agg({"Name": "first", "Sector": "first",
                    "Index": lambda s: " + ".join(sorted(set(s)))}))
    return merged


def full_info(indices: tuple[str, ...] = ("S&P 500",)) -> pd.DataFrame:
    """Stocks + ETFs as one (Name, Sector) table; ETFs flagged via Type column."""
    stocks = stocks_table(indices)
    stocks["Type"] = "Stock"
    etfs = pd.DataFrame(
        [(t, n, s, "ETF") for t, (n, s) in ETFS.items()],
        columns=["Ticker", "Name", "Sector", "Type"],
    ).set_index("Ticker")
    etfs["Index"] = "ETF"
    return pd.concat([stocks, etfs])
