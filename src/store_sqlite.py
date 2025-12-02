"""
Store cleaned CSV into a local SQLite file for testing (no Postgres required).

Usage (PowerShell):
  python src/store_sqlite.py
  python src/store_sqlite.py --csv data/cleaned_reviews.csv --out data/db/reviews.sqlite
"""
from pathlib import Path
import argparse
import sqlite3
import pandas as pd

def main(args):
    csv = Path(args.csv or "data/cleaned_reviews.csv")
    assert csv.exists(), f"CSV not found: {csv}"
    df = pd.read_csv(csv, low_memory=False)
    out = Path(args.out or "data/db/reviews.sqlite")
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))
    # write reviews table
    df.to_sql("reviews", conn, if_exists="replace", index=False)
    # create banks table from distinct values
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS banks AS SELECT DISTINCT bank_code, bank_name FROM reviews;")
    conn.commit()
    # quick verification
    cur.execute("SELECT COUNT(*) FROM reviews;")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM banks;")
    banks = cur.fetchone()[0]
    conn.close()
    print(f"Saved SQLite DB: {out} (reviews={total}, banks={banks})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cleaned_reviews.csv", help="Path to cleaned CSV")
    parser.add_argument("--out", default="data/db/reviews.sqlite", help="Output sqlite file")
    main(parser.parse_args())