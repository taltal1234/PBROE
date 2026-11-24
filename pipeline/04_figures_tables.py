"""Step 4: the figures (results/figures), the LaTeX tables (paper/tables) and paper/numbers.tex.

Every number in the paper is written here from the step 3 outputs, so nothing in the paper
is typed by hand. If a result changes, rerunning this script updates the paper.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so config and src can be imported

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config as C

TEX_DIR = C.PAPER_DIR / "tables"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

WEIGHT_TITLES = {"EW": "Equal weighted", "VW": "Value weighted"}
NAMES = {"s_resid": "P/B ROE residual", "s_k": "Implied return $k$ (two period)", "s_value": "Plain low P/B",
         "s_roe": "ROE alone", "s_resid_sector": "Within sector residual"}
SCATTER_MONTH = "2024-10"    # the month shown in figure 1


# ---------------- small helpers ----------------
def rows(df, **match):
    """The rows of df where every given column equals the given value, e.g. rows(summ, sample="Market")."""
    keep = pd.Series(True, index=df.index)
    for col, value in match.items():
        keep &= df[col] == value
    return df[keep]


def one(df, **match):
    """Like rows(), for when exactly one row is expected."""
    return rows(df, **match).iloc[0]


def f2(x):
    """Two decimals with a real LaTeX minus sign; blank for missing values."""
    return "" if pd.isna(x) else f"{x:.2f}".replace("-", "\\ensuremath{-}")


def tt(x):
    """A t statistic in brackets, as it appears under a coefficient."""
    return "" if pd.isna(x) else f"({f2(x)})"


def write_tabular(name, columns, header, lines, footer=""):
    """Write a booktabs table to paper/tables/<name>. The paper wraps it in its own table environment."""
    body = ("\\begin{tabular}{" + columns + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
            + "\n".join(lines) + footer + "\n\\bottomrule\n\\end{tabular}\n")
    (TEX_DIR / name).write_text(body)


def save_figure(fig, name):
    fig.savefig(C.FIG_DIR / name)
    plt.close(fig)


def month_end_dates(months):
    return pd.PeriodIndex(months, freq="M").to_timestamp() + pd.offsets.MonthEnd(1)


# ---------------- figures ----------------
def figure_scatter(panel):
    """Figure 1: the P/B ROE line in one month."""
    g = panel[panel["month"] == SCATTER_MONTH]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(g["roe_w"], g["lnpb"], s=10, color=BLUE, alpha=0.55, linewidths=0)
    b, a = np.polyfit(g["roe_w"], g["lnpb"], 1)
    x = np.linspace(g["roe_w"].min(), g["roe_w"].max(), 50)
    ax.plot(x, a + b * x, color=ORANGE, lw=2)
    ax.text(x[-1], a + b * x[-1], f"  fitted line, slope {b:.2f}", color=INK, va="center", fontsize=8)
    ax.set_xlabel("ROE (on beginning of year book equity)")
    ax.set_ylabel("ln(P/B)")
    ax.set_title(f"P/B against ROE, PSX firms at end of October 2024 (n = {len(g)})", fontsize=9, color=INK, loc="left")
    save_figure(fig, "fig1_pb_roe_scatter.png")


def figure_quintiles(qm):
    """Figure 2: mean monthly return of each quintile of the main signal."""
    q = rows(qm, sample="Market", signal="s_resid")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, w in zip(axes, ["EW", "VW"]):
        r = q[q["weight"] == w]
        ax.bar(r["portfolio"], r["mean_pct"], color=BLUE, width=0.6)
        for p, v in zip(r["portfolio"], r["mean_pct"]):
            ax.text(p, v + 0.05, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(1, 6), ["1\nexpensive", "2", "3", "4", "5\ncheap"])
        ax.set_title(WEIGHT_TITLES[w], fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean monthly return (%)")
    save_figure(fig, "fig2_quintile_returns.png")


def figure_cumulative(ls):
    """Figure 3: cumulative long short returns, the signal against plain low P/B."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, w in zip(axes, ["EW", "VW"]):
        for sig, color, label in [("s_resid", BLUE, "P/B ROE residual"), ("s_value", ORANGE, "Plain low P/B")]:
            s = rows(ls, sample="Market", signal=sig, weight=w).copy()
            s["date"] = month_end_dates(s["month"])
            s = s.sort_values("date")
            ax.plot(s["date"], np.log1p(s["ls"]).cumsum(), color=color, lw=2, label=label)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.xaxis.set_major_locator(mdates.YearLocator(8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(WEIGHT_TITLES[w], fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("Cumulative log return")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, "fig3_cumulative_ls.png")


def figure_slope(slopes):
    """Figure 4: how much the market pays for ROE, through time."""
    s = slopes[slopes["sample"] == "Market"].copy()
    s["date"] = pd.PeriodIndex(s["month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.plot(s["date"], s["slope"].rolling(12, min_periods=6).mean(), color=BLUE, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Slope on ROE (12 month average)")
    ax.set_title("Cross sectional slope of ln(P/B) on ROE", fontsize=9, color=INK, loc="left")
    save_figure(fig, "fig4_slope_over_time.png")


# ---------------- LaTeX tables ----------------
def table_sample(panel):
    lines = []
    for name, df in [("All firms", panel)] + [(s, panel[panel["sector"] == s]) for s in C.SECTORS]:
        lines.append(f"{name} & {df['gvkey'].nunique()} & {df[df['costat']=='I']['gvkey'].nunique()} & "
                     f"{df.groupby('month')['gvkey'].size().mean():.0f} & {df['pb'].median():.2f} & "
                     f"{df['roe'].median():.3f} & {df['mcap'].median():,.0f} \\\\")
    write_tabular("tab_sample.tex", "lrrrrrr",
                  "Sample & Firms & Delisted & Firms per month & Median P/B & Median ROE & Median size", lines)


def table_quintiles(qm, summ):
    lines = []
    for sig in ["s_resid", "s_value"]:
        for w in ["EW", "VW"]:
            r = rows(qm, sample="Market", signal=sig, weight=w).sort_values("portfolio")
            sm = one(summ, sample="Market", signal=sig, weight=w)
            lines.append(f"{NAMES[sig]}, {w} & " + " & ".join(f2(v) for v in r["mean_pct"]) + f" & {f2(sm['ls_mean_pct'])} \\\\")
            lines.append(" & " + " & ".join(tt(v) for v in r["t"]) + f" & {tt(sm['ls_t'])} \\\\")
    write_tabular("tab_quintiles.tex", "lrrrrrr",
                  "Sort & 1 (expensive) & 2 & 3 & 4 & 5 (cheap) & 5 minus 1", lines)


def table_long_short(summ, alph):
    lines = []
    for sample, sig in [("Market", "s_resid"), ("Market", "s_k"), ("Market", "s_value"), ("Market", "s_roe"),
                        ("Market ex micro", "s_resid"), ("Market ex micro", "s_value")]:
        for w in ["EW", "VW"]:
            sm = one(summ, sample=sample, signal=sig, weight=w)
            ca = one(alph, sample=sample, signal=sig, weight=w, model="CAPM")
            fa = one(alph, sample=sample, signal=sig, weight=w, model="3 factor")
            label = NAMES[sig].replace(" (two period)", "") + (", no microcaps" if "micro" in sample else "")
            lines.append(f"{label} & {w} & {f2(sm['ls_mean_pct'])} & {tt(sm['ls_t'])} & {f2(sm['sharpe_ann'])} & "
                         f"{f2(ca['alpha_pct'])} & {tt(ca['alpha_t'])} & {f2(fa['alpha_pct'])} & {tt(fa['alpha_t'])} & "
                         f"{f2(fa['b_HML'])} \\\\")
    write_tabular("tab_longshort.tex", "llrrrrrrrr",
                  "Signal & Weight & Mean & $t$ & Sharpe & CAPM $\\alpha$ & $t$ & 3F $\\alpha$ & $t$ & HML $\\beta$", lines)


def table_fama_macbeth(fm):
    variables = [("s_resid", "P/B ROE residual (cheapness)"), ("s_value", "ln(B/P)"), ("s_roe", "ROE"),
                 ("lnsize", "ln(size)"), ("mom", "Momentum")]
    specs = ["(1)", "(2)", "(3)", "(4)"]
    lines = []
    for v, label in variables:
        coef_line, t_line = [label], [""]
        for spec in specs:
            r = rows(fm, spec=spec, variable=v)       # empty when the variable is not in that spec
            coef_line.append(f2(r["coef_pct"].iloc[0]) if len(r) else "")
            t_line.append(tt(r["t"].iloc[0]) if len(r) else "")
        lines += [" & ".join(coef_line) + " \\\\", " & ".join(t_line) + " \\\\"]
    months = int(fm["months"].iloc[0])
    footer = f"\n\\midrule\nMonths & {months} & {months} & {months} & {months} \\\\"
    write_tabular("tab_fm.tex", "lrrrr", " & (1) & (2) & (3) & (4)", lines, footer)


def table_sectors(summ, alph, ic):
    lines = []
    for name in C.SECTORS:
        sector_ic = one(ic, sample=name)
        for w in ["EW", "VW"]:
            sm = one(summ, sample=name, weight=w)
            fa = one(alph, sample=name, weight=w, model="3 factor")
            # the rank IC does not depend on the weighting, so it is shown on the EW line only
            ic_cells = f"{sector_ic['mean_ic']:.3f}".replace("-", "\\ensuremath{-}") + f" & {tt(sector_ic['ic_t'])}" if w == "EW" else " & "
            lines.append(f"{name if w == 'EW' else ''} & {w} & {sm['avg_firms']:.0f} & {f2(sm['ls_mean_pct'])} & "
                         f"{tt(sm['ls_t'])} & {f2(fa['alpha_pct'])} & {tt(fa['alpha_t'])} & {ic_cells} \\\\")
    market_ic = one(ic, sample="Market")
    footer = f"\n\\midrule\nWhole market & EW & & & & & & {market_ic['mean_ic']:.3f} & {tt(market_ic['ic_t'])} \\\\"
    write_tabular("tab_sectors.tex", "llrrrrrrr",
                  "Sector & Weight & Firms & LS mean & $t$ & 3F $\\alpha$ & $t$ & Rank IC & $t$", lines, footer)


def table_hit_rates(hit):
    lines = []
    for _, r in hit.iterrows():
        lines.append(f"{r['sample']} & {r['horizon']} & {r['months']} & {r['hit_rate_pct']:.1f} & {r['base_rate_pct']:.1f} & "
                     f"{f2(r['excess_pct'])} & {tt(r['excess_t'])} \\\\")
    write_tabular("tab_hits.tex", "llrrrrr",
                  "Sample & Horizon & Periods & Hit rate (\\%) & Chance rate (\\%) & Excess & $t$", lines)


def signal_label(sample, sig):
    return "Textiles, within sector" if sample == "Textiles" else NAMES[sig]


def table_subperiods(sub):
    lines = []
    for sample, sig in [("Market", "s_resid"), ("Market", "s_value"), ("Textiles", "s_resid_sector")]:
        for w in ["EW", "VW"]:
            r = rows(sub, sample=sample, signal=sig, weight=w)
            cells = " & ".join(f"{f2(x)} & {tt(t)}" for x, t in zip(r["ls_mean_pct"], r["ls_t"]))
            lines.append(f"{signal_label(sample, sig)} & {w} & {cells} \\\\")
    periods = sub["period"].drop_duplicates().tolist()
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{p[:4]} to {p[-7:-3]}}}" for p in periods)
    write_tabular("tab_subperiods.tex", "llrrrrrr", "Signal & Weight & " + head, lines)


def table_costs(cost):
    lines = []
    for sample, sig in [("Market", "s_resid"), ("Textiles", "s_resid_sector"), ("Market", "s_k")]:
        for rebalance in ["monthly", "annual (Oct)"]:
            r = rows(cost, sample=sample, signal=sig, rebalance=rebalance).sort_values("cost_bps")
            cells = " & ".join(f"{f2(x)} & {tt(t)}" for x, t in zip(r["net_ls_mean_pct"], r["t"]))
            lines.append(f"{signal_label(sample, sig)} & {rebalance} & {cells} \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{b} bps}}" for b in C.COST_BPS)
    write_tabular("tab_costs.tex", "llrrrrrr", "Signal & Rebalancing & " + head, lines)


# ---------------- numbers quoted in the text ----------------
def text_numbers(panel, slopes, summ, alph, qm, fm, ic, hit, cost):
    """Every number the paper's text quotes, as name -> formatted string. Written out as LaTeX macros."""
    def ls(sample, sig, w, col):
        return one(summ, sample=sample, signal=sig, weight=w)[col]

    def alpha(sample, sig, w, model, col):
        return one(alph, sample=sample, signal=sig, weight=w, model=model)[col]

    market_slope = slopes[slopes["sample"] == "Market"]["slope"]
    market_ic = one(ic, sample="Market")
    hit_month = one(hit, sample="Market", horizon="1 month")
    hit_year = one(hit, sample="Market", horizon="12 months (Oct)")
    net_100 = one(cost, sample="Market", signal="s_resid", rebalance="monthly", cost_bps=100)
    numbers = {
        "NFirms": panel["gvkey"].nunique(), "NDelisted": panel[panel["costat"] == "I"]["gvkey"].nunique(),
        "NStockMonths": f"{len(panel):,}".replace(",", "{,}"),
        "FirstMonth": pd.Period(panel["month"].min(), "M").strftime("%B %Y"),
        "LastMonth": pd.Period(panel["month"].max(), "M").strftime("%B %Y"),
        "NMonths": panel["month"].nunique(), "SlopeMean": f"{market_slope.mean():.2f}",
        "SlopePos": f"{100 * (market_slope > 0).mean():.1f}",
        "LSew": f2(ls("Market", "s_resid", "EW", "ls_mean_pct")), "LSewT": f2(ls("Market", "s_resid", "EW", "ls_t")),
        "LSvw": f2(ls("Market", "s_resid", "VW", "ls_mean_pct")), "LSvwT": f2(ls("Market", "s_resid", "VW", "ls_t")),
        "LSewAnn": f"{ls('Market', 's_resid', 'EW', 'ls_ann_pct'):.1f}",
        "VALew": f2(ls("Market", "s_value", "EW", "ls_mean_pct")), "VALewT": f2(ls("Market", "s_value", "EW", "ls_t")),
        "VALvw": f2(ls("Market", "s_value", "VW", "ls_mean_pct")), "VALvwT": f2(ls("Market", "s_value", "VW", "ls_t")),
        "FFew": f2(alpha("Market", "s_resid", "EW", "3 factor", "alpha_pct")),
        "FFewT": f2(alpha("Market", "s_resid", "EW", "3 factor", "alpha_t")),
        "FFvw": f2(alpha("Market", "s_resid", "VW", "3 factor", "alpha_pct")),
        "FFvwT": f2(alpha("Market", "s_resid", "VW", "3 factor", "alpha_t")),
        "HMLew": f2(alpha("Market", "s_resid", "EW", "3 factor", "b_HML")),
        "MicroFF": f2(alpha("Market ex micro", "s_resid", "EW", "3 factor", "alpha_pct")),
        "MicroFFT": f2(alpha("Market ex micro", "s_resid", "EW", "3 factor", "alpha_t")),
        "MicroLS": f2(ls("Market ex micro", "s_resid", "EW", "ls_mean_pct")),
        "MicroLST": f2(ls("Market ex micro", "s_resid", "EW", "ls_t")),
        "TEXew": f2(ls("Textiles", "s_resid_sector", "EW", "ls_mean_pct")),
        "TEXewT": f2(ls("Textiles", "s_resid_sector", "EW", "ls_t")),
        "TEXff": f2(alpha("Textiles", "s_resid_sector", "EW", "3 factor", "alpha_pct")),
        "TEXffT": f2(alpha("Textiles", "s_resid_sector", "EW", "3 factor", "alpha_t")),
        "BANKew": f2(ls("Banks", "s_resid_sector", "EW", "ls_mean_pct")),
        "BANKewT": f2(ls("Banks", "s_resid_sector", "EW", "ls_t")),
        "CEMew": f2(ls("Cement", "s_resid_sector", "EW", "ls_mean_pct")),
        "CEMewT": f2(ls("Cement", "s_resid_sector", "EW", "ls_t")),
        "ICmkt": f"{market_ic['mean_ic']:.3f}", "ICmktT": f2(market_ic["ic_t"]),
        "Kew": f2(ls("Market", "s_k", "EW", "ls_mean_pct")), "KewT": f2(ls("Market", "s_k", "EW", "ls_t")),
        "Kvwff": f2(alpha("Market", "s_k", "VW", "3 factor", "alpha_pct")),
        "KvwffT": f2(alpha("Market", "s_k", "VW", "3 factor", "alpha_t")),
        "ROEvwff": f2(alpha("Market", "s_roe", "VW", "3 factor", "alpha_pct")),
        "ROEvwffT": f2(alpha("Market", "s_roe", "VW", "3 factor", "alpha_t")),
        "Turn": f"{100 * ls('Market', 's_resid', 'EW', 'turnover'):.0f}",
        "HitM": f"{hit_month['hit_rate_pct']:.1f}", "BaseM": f"{hit_month['base_rate_pct']:.1f}",
        "HitA": f"{hit_year['hit_rate_pct']:.1f}", "BaseA": f"{hit_year['base_rate_pct']:.1f}",
        "HitAT": f2(hit_year["excess_t"]),
        "NetHundred": f2(net_100["net_ls_mean_pct"]), "NetHundredT": f2(net_100["t"]),
    }

    # how the market factor is described depends on whether the T bill file was there in step 3
    fac = pd.read_parquet(C.PROC_DIR / "factors.parquet")
    numbers["MktDef"] = ("the value weighted return of the universe in excess of the three month Treasury bill yield"
                         if "RF" in fac.columns else "the value weighted return of the universe (no Pakistani risk free "
                         "series is subtracted; because the long short portfolios are self financing this changes only "
                         "the market beta term, by beta times the average bill rate)")

    quintile = rows(qm, sample="Market", signal="s_resid", weight="EW").set_index("portfolio")
    fm4 = fm[fm["spec"] == "(4)"].set_index("variable")
    fm3 = fm[fm["spec"] == "(3)"].set_index("variable")
    numbers.update({
        "QoneEW": f2(quintile.loc[1, "mean_pct"]), "QfiveEW": f2(quintile.loc[C.N_PORT_MARKET, "mean_pct"]),
        "TopEx": f2(ls("Market", "s_resid", "EW", "top_minus_mkt_pct")),
        "TopExT": f2(ls("Market", "s_resid", "EW", "top_minus_mkt_t")),
        "FMfourResid": f2(fm4.loc["s_resid", "coef_pct"]), "FMfourResidT": f2(fm4.loc["s_resid", "t"]),
        "FMfourVal": f2(fm4.loc["s_value", "coef_pct"]), "FMfourValT": f2(fm4.loc["s_value", "t"]),
        "FMthreeROE": f2(fm3.loc["s_roe", "coef_pct"]), "FMthreeROET": f2(fm3.loc["s_roe", "t"]),
    })
    return numbers


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(C.PROC_DIR / "panel.parquet")
    ls = pd.read_parquet(C.PROC_DIR / "ls_returns.parquet")
    slopes = pd.read_parquet(C.PROC_DIR / "slopes.parquet")
    summ, alph, qm, fm, ic, hit, sub, cost = [pd.read_csv(C.TABLE_DIR / f"{name}.csv") for name in
                                              ["ls_summary", "alphas", "portfolio_means", "fama_macbeth",
                                               "rank_ic", "hit_rates", "subperiods", "costs"]]

    figure_scatter(panel)
    figure_quintiles(qm)
    figure_cumulative(ls)
    figure_slope(slopes)

    table_sample(panel)
    table_quintiles(qm, summ)
    table_long_short(summ, alph)
    table_fama_macbeth(fm)
    table_sectors(summ, alph, ic)
    table_hit_rates(hit)
    table_subperiods(sub)
    table_costs(cost)

    numbers = text_numbers(panel, slopes, summ, alph, qm, fm, ic, hit, cost)
    (C.PAPER_DIR / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items()) + "\n")
    print("figures:", sorted(p.name for p in C.FIG_DIR.glob("*.png")))
    print("tables:", sorted(p.name for p in TEX_DIR.glob("*.tex")))
    print("\n".join(f"{k} = {v}" for k, v in numbers.items()))


if __name__ == "__main__":
    main()
