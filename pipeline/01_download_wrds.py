"""Step 1: download the Pakistan data from Compustat Global (WRDS) into data/raw/ (parquet).

In Spyder: open this file and press Run (F5).
    First run with CHECK_ONLY = True  : it only prints what your WRDS account can see.
    Then set CHECK_ONLY = False, Run  : it downloads everything.

Files that already exist in data/raw are skipped, so the download can be stopped and restarted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so config can be imported

import wrds
import config as C

CHECK_ONLY = False

LIB = "comp"
COMPANY = ["gvkey", "conm", "fic", "loc", "sic", "gsector", "gind", "gsubind",
           "costat", "prirow", "dldte", "dlrsn", "ipodate"]
FUNDA = ["gvkey", "datadate", "fyear", "fyr", "curcd", "indfmt", "consol", "popsrc", "datafmt",
         "ceq", "seq", "nicon", "ib", "at", "lt", "sale", "dvt", "dvc"]
SECD = ["gvkey", "iid", "datadate", "prccd", "ajexdi", "trfd", "cshoc", "cshtrd", "curcdd", "exchg"]
IDX = ["gvkeyx", "datadate", "prccd"]
PAKISTAN = f"(fic = '{C.COUNTRY}' or loc = '{C.COUNTRY}')"


def available_columns(db, table, wanted):
    """Keep only the columns this account can see, and say which ones are missing."""
    have = set(db.describe_table(LIB, table)["name"].str.lower())
    missing = [c for c in wanted if c not in have]
    if missing:
        print(f"  comp.{table} is missing: {missing}")
    return [c for c in wanted if c in have]


def already_have(name):
    return (C.RAW_DIR / name).exists()


def save(df, name):
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.RAW_DIR / name, index=False)
    print(f"  saved {name}: {len(df):,} rows")


def quoted_list(values):
    return ", ".join(f"'{v}'" for v in values)


def find_indices(db):
    # a regex instead of LIKE, because the wrds library reads a % sign as a query parameter
    return db.raw_sql("select * from comp.g_idx_index where conm ~* 'karachi|pakistan|kse|psx'")


def check(db):
    tables = set(db.list_tables(library=LIB))
    for t in ["g_company", "g_funda", "g_secd", "g_secm", "g_idx_index", "g_idx_daily"]:
        print(f"comp.{t}:", t in tables)
    available_columns(db, "g_company", COMPANY)
    available_columns(db, "g_funda", FUNDA)
    available_columns(db, "g_secd", SECD)
    n = db.raw_sql(f"select count(*) as n from comp.g_company where {PAKISTAN}")
    print("Pakistani companies in g_company:", int(n["n"][0]))
    if "g_idx_index" in tables:
        print(find_indices(db))


def download(db):
    # 1. every Pakistani company, active and inactive (keeping the dead ones is what removes
    #    survivorship bias). Always fetched again: it is small and the keys are needed below.
    cols = ", ".join(available_columns(db, "g_company", COMPANY))
    firms = db.raw_sql(f"select {cols} from comp.g_company where {PAKISTAN}")
    if not already_have("company.parquet"):
        save(firms, "company.parquet")
    keys = quoted_list(firms["gvkey"])

    # 2. annual fundamentals (consolidated, standard format)
    if not already_have("funda.parquet"):
        cols = ", ".join(available_columns(db, "g_funda", FUNDA))
        sql = f"""select {cols} from comp.g_funda
                  where gvkey in ({keys}) and consol = 'C' and popsrc = 'I' and datafmt = 'HIST_STD'"""
        save(db.raw_sql(sql, date_cols=["datadate"]), "funda.parquet")

    # 3. daily prices, one file per year
    secd = available_columns(db, "g_secd", SECD)
    cols = ", ".join(secd)
    for year in range(C.START_YEAR, C.END_YEAR + 1):
        name = f"prices_{year}.parquet"
        if already_have(name):
            continue
        print("prices", year)
        sql = f"""select {cols} from comp.g_secd
                  where gvkey in ({keys}) and datadate between '{year}-01-01' and '{year}-12-31'"""
        if "curcdd" in secd:
            sql += f" and curcdd = '{C.CURRENCY}'"
        save(db.raw_sql(sql, date_cols=["datadate"]), name)

    # 4. Pakistani market indices (KSE100 and others), if Compustat carries them
    tables = set(db.list_tables(library=LIB))
    if "g_idx_index" in tables and not already_have("index_list.parquet"):
        idx = find_indices(db)
        idx = idx[idx["tici"].str.contains("PAK", na=False)]   # KSE100 and HSBC Pakistan, not the FTSE regional ones
        save(idx, "index_list.parquet")
        if len(idx) and "g_idx_daily" in tables:
            cols = ", ".join(available_columns(db, "g_idx_daily", IDX))
            sql = f"select {cols} from comp.g_idx_daily where gvkeyx in ({quoted_list(idx['gvkeyx'])})"
            save(db.raw_sql(sql, date_cols=["datadate"]), "index_daily.parquet")
    print("ALL DONE")


if __name__ == "__main__":
    db = wrds.Connection()
    if CHECK_ONLY:
        check(db)
    else:
        download(db)
    db.close()
