"""
Store cleaned CSV into a local SQLite file for testing (no Postgres required).
Usage:
  python src\store_sqlite.py --csv  --out data/db/reviews.sqlite
"""
from pathlib import Path
import os, sys
import psycopg2

SQL_FILE = Path("schema.sql")
DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    sys.exit("Set DATABASE_URL environment variable (postgres://user:pass@host:port/dbname)")
if not SQL_FILE.exists():
    sys.exit(f"schema.sql not found at {SQL_FILE.resolve()}")

sql = SQL_FILE.read_text(encoding="utf-8")
print("Connecting to:", DB_URL)
with psycopg2.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
print("Applied schema.sql")

def main(args):
    csv = Path(args.csv)
    assert csv.exists(), f"CSV not found: {csv}"
    df = pd.read_csv(csv, low_memory=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))
    # simple table (columns from CSV)
    df.to_sql("reviews", conn, if_exists="replace", index=False)
    # create simple banks table
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS banks AS SELECT DISTINCT bank_code, bank_name FROM reviews;")
    conn.commit()
    conn.close()
    print(f"Saved SQLite DB: {out} (tables: reviews, banks)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cleaned_reviews.csv")
    parser.add_argument("--out", default="data/db/reviews.sqlite")
    main(parser.parse_args())