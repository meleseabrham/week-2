"""
Store cleaned CSV to PostgreSQL using schema required by Task 3.
Usage:
  set DATABASE_URL (postgres://user:pass@host:port/dbname)
  python src\store_postgres.py --csv data\cleaned_reviews.csv
"""
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch, Json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CREATE_BANKS = """
CREATE TABLE IF NOT EXISTS banks (
  id SERIAL PRIMARY KEY,
  bank_code TEXT UNIQUE NOT NULL,
  bank_name TEXT
);
"""

CREATE_REVIEWS = """
CREATE TABLE IF NOT EXISTS reviews (
  id SERIAL PRIMARY KEY,
  review_id TEXT UNIQUE NOT NULL,
  bank_code TEXT NOT NULL REFERENCES banks(bank_code),
  rating NUMERIC,
  review_date TIMESTAMP,
  source TEXT,
  content TEXT,
  sentiment TEXT,
  metadata JSONB
);
"""

def connect(conn_uri: str):
    return psycopg2.connect(conn_uri)

def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_BANKS)
        cur.execute(CREATE_REVIEWS)
    conn.commit()
    logging.info("Ensured PostgreSQL schema (banks, reviews)")

def upsert_banks(conn, bank_codes):
    with conn.cursor() as cur:
        for code in bank_codes:
            cur.execute("INSERT INTO banks (bank_code) VALUES (%s) ON CONFLICT (bank_code) DO NOTHING", (code,))
    conn.commit()
    logging.info("Upserted %d banks", len(bank_codes))

def insert_reviews(conn, rows):
    sql = """
    INSERT INTO reviews (review_id, bank_code, rating, review_date, source, content, sentiment, metadata)
    VALUES (%(review_id)s, %(bank_code)s, %(rating)s, %(date)s, %(source)s, %(content)s, %(sentiment)s, %(metadata)s)
    ON CONFLICT (review_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=200)
    conn.commit()
    logging.info("Inserted batch of %d reviews", len(rows))

def verify_counts(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        rcount = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM banks")
        bcount = cur.fetchone()[0]
    logging.info("Postgres verification: banks=%d reviews=%d", bcount, rcount)
    return bcount, rcount

def main(args: argparse.Namespace):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"], low_memory=False)
    # minimal normalization to match schema
    df = df.rename(columns={"date":"date","bank_code":"bank_code","review_id":"review_id","content":"content","rating":"rating","source":"source"})
    df["sentiment"] = df.get("sentiment")  # if produced earlier
    # prepare connection
    db = os.environ.get("DATABASE_URL")
    if not db:
        raise SystemExit("Set DATABASE_URL env var (postgres://user:pass@host:port/db)")
    conn = connect(db)
    ensure_schema(conn)
    bank_codes = df["bank_code"].dropna().unique().tolist()
    upsert_banks(conn, bank_codes)
    # prepare rows
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "review_id": str(r.get("review_id")),
            "bank_code": str(r.get("bank_code")),
            "rating": None if pd.isna(r.get("rating")) else float(r.get("rating")),
            "date": None if pd.isna(r.get("date")) else r.get("date").to_pydatetime() if hasattr(r.get("date"), "to_pydatetime") else r.get("date"),
            "source": r.get("source"),
            "content": r.get("content"),
            "sentiment": r.get("sentiment", None),
            "metadata": Json({"ingested_by":"store_postgres.py"})
        })
        # Insert in batches of 1000
        if len(rows) >= 1000:
            insert_reviews(conn, rows)
            rows = []
    if rows:
        insert_reviews(conn, rows)
    verify_counts(conn)
    conn.close()
    logging.info("Done: stored CSV into PostgreSQL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV file path", default="data/cleaned_reviews.csv")
    args = parser.parse_args()
    main(args)

"""
Apply SQL schema.sql to the PostgreSQL database using DATABASE_URL env var.

Usage (PowerShell):
  $env:DATABASE_URL="postgres://user:pass@host:5432/dbname"
  python src\apply_schema.py
"""
import os
from pathlib import Path
import psycopg2

SQL_FILE = Path("schema.sql")
DB_URL = os.environ.get("DATABASE_URL")

if not DB_URL:
    raise SystemExit("Set DATABASE_URL environment variable (postgres://user:pass@host:port/dbname)")

if not SQL_FILE.exists():
    raise SystemExit(f"schema.sql not found at {SQL_FILE.resolve()}")

sql = SQL_FILE.read_text(encoding="utf-8")

print("Connecting to:", DB_URL)
with psycopg2.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
print("Applied schema.sql")