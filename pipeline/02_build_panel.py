"""Step 2: build the monthly stock panel and the P/B ROE signals from data/raw.

Outputs (data/processed/)
    panel.parquet    one row per stock month, with the signals
    slopes.parquet   the monthly slope of ln(P/B) on ROE, for the market and each sector
    market.parquet   monthly KSE100 returns

Timing is the thing to get right here. A firm's annual numbers are only used from 4 months
after its fiscal year end (about when the annual report is public), and for at most 18 months
after the year end. A signal measured at the end of month t is compared with the return in t+1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so config and src can be imported

import numpy as np
import pandas as pd
import config as C
from src.utils import load, months_apart
from src.stats import winsorize, ols_residuals

FUNDAMENTALS = ["book", "ib", "roe", "payout", "at"]
PANEL_COLUMNS = ["gvkey", "conm", "month", "sector", "gsector", "costat", "fyr", "prccd", "mcap", "book",
                 "roe", "roe_w", "payout", "pb", "lnpb", "lnsize", "mom", "days_traded", "ret", "ret_next",
                 "resid", "resid_sector", "k_implied"]


def sector_of(firm):
    """Banks, Cement, Textiles or Other, by GICS sub industry first and SIC code second."""
    gics = str(firm["gsubind"]) if pd.notna(firm["gsubind"]) else ""
    sic = int(firm["sic"]) if pd.notna(firm["sic"]) else 0
    for name, rule in C.SECTORS.items():
        if gics.startswith(rule["gics"]) or rule["sic"][0] <= sic <= rule["sic"][1]:
            return name
    return "Other"


def load_firms():
    firms = load(C.RAW_DIR / "company.parquet")
    firms["sector"] = firms.apply(sector_of, axis=1)
    firms["gsector"] = firms["gsector"].fillna("NA")
    firms["gvkey"] = firms["gvkey"].astype(str)
    return firms


def monthly_returns():
    """Month end price, market cap and total return of every stock, from the daily file."""
    px = pd.concat([load(f) for f in sorted(C.RAW_DIR.glob("prices_*.parquet"))])

    # a few firms list more than one share class: keep the one with the longest price history
    n_days = px.groupby(["gvkey", "iid"]).size().reset_index(name="n")
    main_class = n_days.sort_values("n").drop_duplicates("gvkey", keep="last")
    px = px.merge(main_class[["gvkey", "iid"]], on=["gvkey", "iid"])

    px["month"] = px["datadate"].dt.to_period("M")
    px["traded"] = (px["cshtrd"] > 0).astype(int)
    px["tri"] = px["prccd"] / px["ajexdi"] * px["trfd"]          # total return index (splits and dividends)
    px = px.sort_values(["gvkey", "datadate"])
    days = px.groupby(["gvkey", "month"])["traded"].sum().rename("days_traded")
    px = px[px["month"] <= pd.Period(C.LAST_FULL_MONTH, "M")]   # the current month is not over yet

    m = px.groupby(["gvkey", "month"]).tail(1).set_index(["gvkey", "month"])
    m = m[["prccd", "cshoc", "tri"]].join(days).reset_index()
    m["mcap"] = m["prccd"] * m["cshoc"] / 1e6                    # PKR millions, the same unit as book equity

    # a return only counts if the previous month end is really the month before (no gaps)
    m = m.sort_values(["gvkey", "month"])
    prev_tri = m.groupby("gvkey")["tri"].shift(1)
    prev_month = m.groupby("gvkey")["month"].shift(1)
    next_month = m.groupby("gvkey")["month"].shift(-1)
    consecutive = months_apart(m["month"], prev_month) == 1
    m["ret"] = np.where(consecutive, m["tri"] / prev_tri - 1, np.nan)
    m.loc[m["ret"] > 3, "ret"] = np.nan                          # more than +300% in a month is a data error
    m["ret_next"] = m.groupby("gvkey")["ret"].shift(-1)
    m.loc[months_apart(next_month, m["month"]) != 1, "ret_next"] = np.nan
    m["gvkey"] = m["gvkey"].astype(str)
    return m


def annual_fundamentals():
    """Book equity, earnings and ROE per firm year, with the month they become public."""
    f = load(C.RAW_DIR / "funda.parquet")
    f = f[f["curcd"] == C.CURRENCY].copy()
    f["book"] = f["ceq"].fillna(f["seq"])
    f = f.sort_values(["gvkey", "datadate"])

    # ROE uses book equity at the start of the year, so the previous report must be about a year back
    f["book_lag"] = f.groupby("gvkey")["book"].shift(1)
    gap = (f["datadate"] - f.groupby("gvkey")["datadate"].shift(1)).dt.days
    f.loc[(gap < 300) | (gap > 430), "book_lag"] = np.nan
    f["roe"] = f["ib"] / f["book_lag"].where(f["book_lag"] > 0)
    f["payout"] = f["dvc"] / f["ib"].where(f["ib"] > 0)

    f["avail"] = (f["datadate"] + pd.DateOffset(months=4)).dt.to_period("M")
    f = f[["gvkey", "datadate", "fyr", "avail"] + FUNDAMENTALS]
    f["adate"] = f["avail"].dt.to_timestamp().astype("datetime64[ns]")
    f["gvkey"] = f["gvkey"].astype(str)
    return f


def merge_panel(m, f, firms):
    """Attach to every stock month the latest fundamentals that were public by then."""
    m["mdate"] = m["month"].dt.to_timestamp().astype("datetime64[ns]")
    panel = pd.merge_asof(m.sort_values("mdate"), f.sort_values("adate"), left_on="mdate", right_on="adate",
                          by="gvkey", direction="backward")
    # public 4 months after the year end, so 14 more months takes us to 18 months after it
    age = (panel["mdate"] - panel["adate"]).dt.days
    panel.loc[age > 14 * 31, FUNDAMENTALS] = np.nan
    panel = panel.merge(firms[["gvkey", "conm", "sector", "gsector", "costat"]], on="gvkey", how="left")

    panel["pb"] = panel["mcap"] / panel["book"].where(panel["book"] > 0)
    panel["lnpb"] = np.log(panel["pb"])
    panel["lnsize"] = np.log(panel["mcap"])
    # momentum: cumulative return from t minus 11 to t minus 1 (the latest month is skipped)
    panel["lr"] = np.log1p(panel["ret"])
    panel["mom"] = np.expm1(panel.groupby("gvkey")["lr"].transform(
        lambda s: s.shift(1).rolling(11, min_periods=8).sum()))

    ok = (panel["book"] > 0) & panel["roe"].notna() & (panel["mcap"] > 0) & panel["lnpb"].notna()
    ok &= panel["days_traded"] >= C.MIN_DAYS_TRADED
    ok &= panel["prccd"] >= C.MIN_PRICE
    panel["in_universe"] = ok
    return panel


def monthly_signals(panel):
    """Run the P/B ROE regressions month by month. The residual is the signal:
    a stock below the line is cheap for the ROE it earns."""
    months, slopes = [], []
    for month, g in panel[panel["in_universe"]].groupby("month"):
        if len(g) < C.MIN_FIRMS_MARKET:
            continue
        g = g.copy()
        g["roe_w"] = winsorize(g["roe"])

        # main signal: whole market, one ROE slope, its own intercept for each GICS sector
        # (sectors with fewer than 3 firms share one intercept, otherwise they fit perfectly)
        small = g["gsector"].map(g["gsector"].value_counts()) < 3
        g["grp"] = np.where(small, "small", g["gsector"])
        g["resid"], slope = ols_residuals(g, "lnpb", ["roe_w"], dummies="grp")
        slopes.append({"month": month, "sample": "Market", "slope": slope, "n": len(g)})

        # Wilcox and Philips two period form: implied expected return k = ROE minus residual / T
        g["k_implied"] = g["roe_w"] - g["resid"] / C.HORIZON_YEARS

        # sector signals: a separate regression inside each featured sector
        g["resid_sector"] = np.nan
        for name in C.SECTORS:
            s = g[g["sector"] == name]
            if len(s) >= C.MIN_FIRMS_SECTOR:
                e, slope = ols_residuals(s, "lnpb", ["roe_w"])
                g.loc[s.index, "resid_sector"] = e
                slopes.append({"month": month, "sample": name, "slope": slope, "n": len(s)})
        months.append(g)

    sig = pd.concat(months)[PANEL_COLUMNS]
    sig["month"] = sig["month"].astype(str)
    slopes = pd.DataFrame(slopes)
    slopes["month"] = slopes["month"].astype(str)
    return sig, slopes


def kse100_returns():
    idx = load(C.RAW_DIR / "index_daily.parquet")
    idx = idx[idx["gvkeyx"] == C.KSE100_GVKEYX].sort_values("datadate")
    idx["month"] = idx["datadate"].dt.to_period("M")
    k = idx.groupby("month").tail(1)[["month", "prccd"]]
    k["kse100_ret"] = k["prccd"].pct_change()
    k["month"] = k["month"].astype(str)
    return k[["month", "kse100_ret"]]


def main():
    panel = merge_panel(monthly_returns(), annual_fundamentals(), load_firms())
    sig, slopes = monthly_signals(panel)

    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(C.PROC_DIR / "panel.parquet", index=False)
    slopes.to_parquet(C.PROC_DIR / "slopes.parquet", index=False)
    kse100_returns().to_parquet(C.PROC_DIR / "market.parquet", index=False)

    print("stock months in universe:", len(sig), " firms:", sig["gvkey"].nunique(),
          " months:", sig["month"].nunique(), sig["month"].min(), "to", sig["month"].max())
    print(sig.groupby(sig["month"].str[:4]).agg(firms=("gvkey", "nunique")).T.to_string())


if __name__ == "__main__":
    main()
