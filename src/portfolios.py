"""Portfolio sorts. Formed at the end of month t, held over month t+1."""
import numpy as np
import pandas as pd


def assign_portfolios(scores, n_port):
    """Portfolio number 1 (lowest score) to n_port (highest score).
    Ranking first breaks ties, so every portfolio gets the same number of stocks."""
    return pd.qcut(scores.rank(method="first"), n_port, labels=False) + 1


def sort_portfolios(df, score, n_port):
    """Monthly equal and value weighted returns of n_port portfolios sorted on score.

    Also keeps track of turnover in the two legs: the share of names in the top (and bottom)
    portfolio that were not there the month before. The long short return is top minus bottom.
    """
    rows, previous = [], {}
    for month, g in df.groupby("month"):
        g = g[g[score].notna()]
        if len(g) < 3 * n_port:
            continue
        p = assign_portfolios(g[score], n_port)
        row = {"month": month, "n": len(g)}
        for k in range(1, n_port + 1):
            h = g[p == k]
            row[f"ew{k}"] = h["ret_next"].mean()
            row[f"vw{k}"] = np.average(h["ret_next"], weights=h["mcap"])
        for leg, k in [("top", n_port), ("bot", 1)]:
            members = set(g.loc[p == k, "gvkey"])
            old = previous.get(leg)
            row[f"to_{leg}"] = np.nan if old is None else 1 - len(members & old) / max(len(members), 1)
            previous[leg] = members
        row["ew_all"] = g["ret_next"].mean()
        rows.append(row)
    r = pd.DataFrame(rows).set_index("month")
    r["ls_ew"] = r[f"ew{n_port}"] - r["ew1"]
    r["ls_vw"] = r[f"vw{n_port}"] - r["vw1"]
    return r


def book_turnover(r):
    """Share of the whole long short book replaced each month (both legs added up)."""
    return r["to_top"] + r["to_bot"]


def annual_long_short(df, score, n_port):
    """Long short return when the portfolios are formed every October and then held for 12 months.
    Stocks are equally weighted inside each leg every month."""
    rows = []
    formation = df[df["month"].dt.month == 10]
    for form_month, g in formation.groupby("month"):
        g = g[g[score].notna()]
        if len(g) < 3 * n_port:
            continue
        p = assign_portfolios(g[score], n_port)
        top, bot = set(g.loc[p == n_port, "gvkey"]), set(g.loc[p == 1, "gvkey"])
        hold = df[(df["month"] >= form_month) & (df["month"] < form_month + 12)]
        for month, h in hold.groupby("month"):
            rows.append({"month": month, "ls": h.loc[h["gvkey"].isin(top), "ret_next"].mean()
                         - h.loc[h["gvkey"].isin(bot), "ret_next"].mean()})
    return pd.DataFrame(rows).set_index("month")["ls"]
