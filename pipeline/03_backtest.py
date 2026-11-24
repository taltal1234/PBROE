"""Step 3: portfolio tests of the P/B ROE signals.

Every signal is first turned into a "cheapness" score, where higher means cheaper. Portfolios
are formed at the end of month t and held over month t+1, and the long short (LS) return is the
cheapest portfolio minus the dearest one. Tables go to results/tables, return series to data/processed.

The tests, in the order they appear below:
    1. portfolio sorts (market, market without microcaps, each sector)
    2. rank IC and hit rates against the chance rate
    3. Fama and MacBeth regressions
    4. size and value factors built from the same universe, and CAPM / 3 factor alphas
    5. subperiods
    6. trading costs, monthly against annual rebalancing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so config and src can be imported

import numpy as np
import pandas as pd
import config as C
from src.utils import save_table
from src.stats import newey_west, ols_newey_west, zscore, hit_and_chance
from src.portfolios import sort_portfolios, book_turnover, annual_long_short

# (sample, score, residual it comes from) for the IC and hit rate tests
RESIDUAL_SIGNALS = [("Market", "s_resid", "resid")] + [(s, "s_resid_sector", "resid_sector") for s in C.SECTORS]
MIN_FIRMS_IC = 8


def load_signals():
    d = pd.read_parquet(C.PROC_DIR / "panel.parquet")
    d["month"] = pd.PeriodIndex(d["month"], freq="M")
    d["s_resid"] = -d["resid"]                 # main signal: below the P/B ROE line
    d["s_resid_sector"] = -d["resid_sector"]   # below the line of its own sector
    d["s_k"] = d["k_implied"]                  # two period implied expected return
    d["s_value"] = -d["lnpb"]                  # plain low P/B, the benchmark the signal has to beat
    d["s_roe"] = d["roe_w"]                    # profitability on its own

    # forward 12 month return for the annual tests (compounded, needs at least 6 months of data)
    d = d.sort_values(["gvkey", "month"])
    d["lr_next"] = np.log1p(d["ret_next"])
    d["fwd12"] = np.expm1(d.groupby("gvkey")["lr_next"].transform(
        lambda s: s[::-1].rolling(12, min_periods=6).sum()[::-1]))
    return d


def in_sample(df, sample):
    return df if sample == "Market" else df[df["sector"] == sample]


def sort_tests(ev):
    """Every (sample, score, number of portfolios, data) combination that gets a portfolio sort."""
    tests = [("Market", score, C.N_PORT_MARKET, ev) for score in ["s_resid", "s_k", "s_value", "s_roe"]]
    # robustness: drop microcaps (the bottom 30% of market cap each month)
    cut = ev.groupby("month")["mcap"].transform(lambda s: s.quantile(C.MICROCAP_PCT))
    ev_large = ev[ev["mcap"] >= cut]
    tests += [("Market ex micro", score, C.N_PORT_MARKET, ev_large) for score in ["s_resid", "s_value"]]
    tests += [(name, "s_resid_sector", C.N_PORT_SECTOR, in_sample(ev, name)) for name in C.SECTORS]
    return tests


# ---------------- 1. portfolio sorts ----------------
def long_short_summary(tests):
    port, summary, series = {}, [], []
    for sample, score, n_port, df in tests:
        r = sort_portfolios(df, score, n_port)
        port[(sample, score)] = r
        for w in C.WEIGHTS:
            ls = r[f"ls_{w}"]
            mu, t, n = newey_west(ls)
            top_ex, top_t, _ = newey_west(r[f"{w}{n_port}"] - r["ew_all"])   # cheap leg against the average stock
            summary.append({"sample": sample, "signal": score, "weight": w.upper(), "portfolios": n_port,
                            "months": n, "first": str(r.index.min()), "avg_firms": round(r["n"].mean()),
                            "ls_mean_pct": 100 * mu, "ls_t": t, "ls_ann_pct": 100 * 12 * mu,
                            "sharpe_ann": ls.mean() / ls.std() * np.sqrt(12),
                            "pct_months_pos": 100 * (ls > 0).mean(),
                            "top_minus_mkt_pct": 100 * top_ex, "top_minus_mkt_t": top_t,
                            "turnover": book_turnover(r).mean()})
            s = ls.rename("ls").to_frame()
            s["sample"], s["signal"], s["weight"] = sample, score, w.upper()
            series.append(s.reset_index())

    summary = save_table(summary, "ls_summary", 3)
    ls = pd.concat(series)
    ls["month"] = ls["month"].astype(str)
    ls.to_parquet(C.PROC_DIR / "ls_returns.parquet", index=False)
    return port, summary


def portfolio_means(port):
    rows = []
    for (sample, score), r in port.items():
        n_port = C.N_PORT_SECTOR if sample in C.SECTORS else C.N_PORT_MARKET
        for k in range(1, n_port + 1):
            for w in C.WEIGHTS:
                mu, t, _ = newey_west(r[f"{w}{k}"])
                rows.append({"sample": sample, "signal": score, "weight": w.upper(), "portfolio": k,
                             "mean_pct": 100 * mu, "t": t})
    save_table(rows, "portfolio_means", 3)


# ---------------- 2. rank IC and hit rate ----------------
def hit_rate_row(sample, horizon, hits, lags):
    h = pd.DataFrame(hits)
    mu, t, n = newey_west(h["hit"] - h["base"], lags=lags)
    return {"sample": sample, "horizon": horizon, "months": n, "hit_rate_pct": 100 * h["hit"].mean(),
            "base_rate_pct": 100 * h["base"].mean(), "excess_pct": 100 * mu, "excess_t": t}


def rank_ic_and_hit_rates(ev, d):
    ic_rows, hit_rows = [], []
    for sample, score, resid_col in RESIDUAL_SIGNALS:
        ics, hits = [], []
        for month, g in in_sample(ev, sample).groupby("month"):
            g = g[g[score].notna()]
            if len(g) < MIN_FIRMS_IC:
                continue
            ics.append(g[score].rank().corr(g["ret_next"].rank()))
            hits.append(hit_and_chance(g[resid_col] < 0, g["ret_next"] > g["ret_next"].mean()))
        mu, t, n = newey_west(ics)
        ic_rows.append({"sample": sample, "months": n, "mean_ic": mu, "ic_t": t,
                        "pct_months_pos": 100 * np.mean(np.array(ics) > 0)})
        hit_rows.append(hit_rate_row(sample, "1 month", hits, lags=C.NW_LAGS))

    # the annual version of the test in my senior project: signal at the end of October,
    # compared with the next 12 months (one observation a year, so only 1 lag)
    for sample, _, resid_col in RESIDUAL_SIGNALS:
        df = d[(d["month"].dt.month == 10) & d["fwd12"].notna() & d[resid_col].notna()]
        hits = []
        for month, g in in_sample(df, sample).groupby("month"):
            if len(g) < MIN_FIRMS_IC:
                continue
            hits.append(hit_and_chance(g[resid_col] < 0, g["fwd12"] > g["fwd12"].mean()))
        hit_rows.append(hit_rate_row(sample, "12 months (Oct)", hits, lags=1))

    save_table(ic_rows, "rank_ic", 4)
    save_table(hit_rows, "hit_rates", 3)


# ---------------- 3. Fama and MacBeth ----------------
FM_SPECS = {"(1)": ["s_resid"], "(2)": ["s_resid", "lnsize", "mom"],
            "(3)": ["s_value", "s_roe", "lnsize", "mom"], "(4)": ["s_resid", "s_value", "lnsize", "mom"]}


def fama_macbeth(ev):
    """One cross sectional regression per month on standardized characteristics, then the
    time series average of each coefficient with a Newey West t statistic."""
    rows = []
    for spec, xs in FM_SPECS.items():
        coefs = []
        for month, g in ev.groupby("month"):
            g = g.dropna(subset=xs + ["ret_next"])
            if len(g) < C.MIN_FIRMS_MARKET:
                continue
            X = np.column_stack([np.ones(len(g))] + [zscore(g[x]).to_numpy() for x in xs])
            coefs.append(np.linalg.lstsq(X, g["ret_next"].to_numpy(), rcond=None)[0])
        coefs = np.array(coefs)
        for j, x in enumerate(["const"] + xs):
            mu, t, n = newey_west(coefs[:, j])
            rows.append({"spec": spec, "variable": x, "coef_pct": 100 * mu, "t": t, "months": n})
    save_table(rows, "fama_macbeth", 3)


# ---------------- 4. factors and alphas ----------------
def build_factors(ev):
    """Fama French style MKT, SMB and HML from the same Pakistani universe:
    2 size groups (median) by 3 value groups (30th and 70th percentiles), value weighted."""
    rows = []
    for month, g in ev.groupby("month"):
        g = g.dropna(subset=["s_value"])
        big = g["mcap"] >= g["mcap"].median()
        lo, hi = g["s_value"].quantile([0.3, 0.7])
        value_group = np.where(g["s_value"] <= lo, "L", np.where(g["s_value"] >= hi, "H", "M"))
        vw = {}
        for size, in_size in [("B", big), ("S", ~big)]:
            for v in ["L", "M", "H"]:
                h = g[in_size & (value_group == v)]
                vw[size + v] = np.average(h["ret_next"], weights=h["mcap"]) if len(h) else np.nan
        rows.append({"month": month, "MKT": np.average(g["ret_next"], weights=g["mcap"]),
                     "SMB": (vw["SL"] + vw["SM"] + vw["SH"]) / 3 - (vw["BL"] + vw["BM"] + vw["BH"]) / 3,
                     "HML": (vw["SH"] + vw["BH"]) / 2 - (vw["SL"] + vw["BL"]) / 2})
    return pd.DataFrame(rows).set_index("month")


def subtract_risk_free(fac):
    """Turn MKT into an excess return with the SBP 3 month T bill auction yield.

    The SBP file has one row per auction; I read the "Old Format" sheet (1998 onward), average the
    3 month weighted average yield by month and carry it forward over months with no auction.
    """
    rf_file = C.RAW_DIR / "sbp_tb_auctions.xlsx"
    if not rf_file.exists():
        return "no risk free file found: MKT is the raw market return"
    tb = pd.read_excel(rf_file, sheet_name="Old Format", header=None)
    tb = tb[tb[1].apply(lambda v: hasattr(v, "year"))].copy()   # auction rows have a date in column B
    tb["month"] = pd.to_datetime(tb[1].tolist()).to_period("M")
    tb["y3m"] = pd.to_numeric(tb[4], errors="coerce")            # annual decimal yield; text means no bids
    rf = tb.groupby("month")["y3m"].mean()
    rf = rf.reindex(pd.period_range(rf.index.min(), rf.index.max(), freq="M")).ffill() / 12
    fac["RF"] = rf.reindex(fac.index).to_numpy()                 # yield known in month t, earned over t+1
    fac["MKT"] = fac["MKT"] - fac["RF"]
    return f"MKT is in excess of the SBP 3 month T bill yield; RF months used: {fac['RF'].notna().sum()} of {len(fac)}"


def factor_alphas(port, fac):
    rows = []
    for (sample, score), r in port.items():
        for w in C.WEIGHTS:
            j = r[[f"ls_{w}"]].join(fac[["MKT", "SMB", "HML"]], how="inner").dropna()
            for model, xs in [("CAPM", ["MKT"]), ("3 factor", ["MKT", "SMB", "HML"])]:
                b, t = ols_newey_west(j[f"ls_{w}"].to_numpy(), j[xs].to_numpy())
                row = {"sample": sample, "signal": score, "weight": w.upper(), "model": model,
                       "alpha_pct": 100 * b[0], "alpha_t": t[0], "months": len(j)}
                for x, bb, tt in zip(xs, b[1:], t[1:]):
                    row[f"b_{x}"], row[f"t_{x}"] = bb, tt
                rows.append(row)
    save_table(rows, "alphas", 3)


# ---------------- 5. subperiods ----------------
def subperiods(port):
    rows = []
    for (sample, score), r in port.items():
        for start, end in C.SUBPERIODS:
            part = r[(r.index >= pd.Period(start, "M")) & (r.index <= pd.Period(end, "M"))]
            for w in C.WEIGHTS:
                if len(part) < 12:
                    continue
                mu, t, n = newey_west(part[f"ls_{w}"])
                rows.append({"sample": sample, "signal": score, "weight": w.upper(), "period": f"{start} to {end}",
                             "months": n, "ls_mean_pct": 100 * mu, "ls_t": t})
    save_table(rows, "subperiods", 3)


# ---------------- 6. trading costs ----------------
def trading_costs(tests, port):
    """Net LS return after costs. Monthly: every replaced name is one sale and one purchase.
    Annual: the whole book (both legs) is replaced once a year, in October."""
    rows = []
    for sample, score, n_port, df in tests:
        if score not in ["s_resid", "s_resid_sector", "s_k"]:
            continue
        r = port[(sample, score)]
        turn = book_turnover(r).fillna(1)        # first month: everything is bought
        annual = annual_long_short(df, score, n_port)
        for bps in C.COST_BPS:
            c = bps / 1e4
            mu, t, n = newey_west(r["ls_ew"] - 2 * c * turn)
            rows.append({"sample": sample, "signal": score, "rebalance": "monthly", "cost_bps": bps,
                         "net_ls_mean_pct": 100 * mu, "t": t, "months": n})
            net_annual = annual - np.where(annual.index.month == 10, 2 * 2 * c, 0)
            mu, t, n = newey_west(pd.Series(net_annual))
            rows.append({"sample": sample, "signal": score, "rebalance": "annual (Oct)", "cost_bps": bps,
                         "net_ls_mean_pct": 100 * mu, "t": t, "months": n})
    save_table(rows, "costs", 3)


def main():
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    d = load_signals()
    ev = d[d["ret_next"].notna()].copy()     # evaluation sample: the next month's return is known
    tests = sort_tests(ev)

    port, summary = long_short_summary(tests)
    portfolio_means(port)
    rank_ic_and_hit_rates(ev, d)
    fama_macbeth(ev)

    fac = build_factors(ev)
    rf_note = subtract_risk_free(fac)
    fac["month_str"] = fac.index.astype(str)
    fac.reset_index(drop=True).to_parquet(C.PROC_DIR / "factors.parquet", index=False)
    factor_alphas(port, fac)

    subperiods(port)
    trading_costs(tests, port)

    print(rf_note)
    print(summary[["sample", "signal", "weight", "months", "avg_firms", "ls_mean_pct", "ls_t", "sharpe_ann",
                   "top_minus_mkt_pct", "top_minus_mkt_t", "turnover"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
