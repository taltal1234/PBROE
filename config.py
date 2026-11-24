"""Every setting for the P/B ROE Pakistan project.

If you want to change the sample, a filter or a test, this is the only file you should need to touch.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"            # WRDS downloads, never edited by hand
PROC_DIR = ROOT / "data" / "processed"     # panels built from raw
TABLE_DIR = ROOT / "results" / "tables"    # csv tables
FIG_DIR = ROOT / "results" / "figures"     # png figures
PAPER_DIR = ROOT / "paper"                 # the paper and its generated tables

# ---------------- sample ----------------
COUNTRY = "PAK"               # Compustat Global country code (fic or loc)
CURRENCY = "PKR"
START_YEAR = 1995             # first year of daily prices to download
END_YEAR = 2026               # last year of daily prices to download
LAST_FULL_MONTH = "2026-10"   # last complete month of returns (the data were pulled mid November 2026)

# ---------------- sectors studied on their own ----------------
# A firm is matched on its GICS code first. The SIC range is only a fallback for firms with no GICS code.
SECTORS = {
    "Banks":    {"gics": "401010",   "sic": (6020, 6029)},
    "Cement":   {"gics": "15102010", "sic": (3241, 3241)},
    "Textiles": {"gics": "252030",   "sic": (2200, 2299)},
}

# ---------------- universe and signal ----------------
MIN_DAYS_TRADED = 5      # a stock must trade on at least this many days in the formation month
MIN_PRICE = 1.0          # PKR, drops penny stocks
MIN_FIRMS_MARKET = 30    # months with fewer usable firms are skipped
MIN_FIRMS_SECTOR = 8     # a sector regression needs at least this many firms
HORIZON_YEARS = 5        # T in the two period Wilcox and Philips model
KSE100_GVKEYX = "150032" # Compustat Global id of the KSE 100 index

# ---------------- portfolios and tests ----------------
N_PORT_MARKET = 5        # quintiles for the whole market
N_PORT_SECTOR = 3        # terciles inside a sector (sectors are small)
MICROCAP_PCT = 0.30      # robustness check: drop stocks below this market cap percentile
NW_LAGS = 6              # Newey West lags for the monthly t statistics
COST_BPS = [0, 50, 100]  # one way trading cost scenarios, in basis points
SUBPERIODS = [("1998-01", "2007-12"), ("2008-01", "2016-12"), ("2017-01", "2026-12")]
WEIGHTS = ["ew", "vw"]   # every portfolio is reported equal weighted and value weighted
