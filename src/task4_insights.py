import logging
from pathlib import Path
from collections import Counter
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sns.set(style="whitegrid")

IN = Path("data/cleaned_reviews.csv")
OUT_DIR = Path("data/analysis")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "the","and","for","with","that","this","have","has","was","but","not","you","are","app","bank","good",
    "very","get","like","one","use","using","service","account"
}

def clean_tokens(text: str):
    t = re.sub(r"[^A-Za-z0-9\s']", " ", str(text)).lower()
    return [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]

def sentiment_blob(text: str, thresh=0.1):
    b = TextBlob(str(text))
    p = b.sentiment.polarity
    if p > thresh: return "positive", p
    if p < -thresh: return "negative", p
    return "neutral", p

def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"], df["sentiment_score"] = zip(*df["content"].fillna("").map(lambda t: sentiment_blob(t)))
    return df

def coverage_check(df: pd.DataFrame, min_coverage=0.90):
    total = len(df)
    scored = df["sentiment"].notna().sum()
    cov = scored / max(1, total)
    logging.info("Sentiment coverage: %d/%d (%.1f%%)", scored, total, cov*100)
    if cov < min_coverage:
        logging.warning("Coverage below %.0f%%", min_coverage*100)
    return cov

def per_bank_drivers(df: pd.DataFrame, topn=5):
    banks = {}
    for bank, g in df.groupby("bank_code"):
        pos_text = " ".join(g[g["sentiment"]=="positive"]["content"].dropna().astype(str).values)
        neg_text = " ".join(g[g["sentiment"]=="negative"]["content"].dropna().astype(str).values)
        pos_words = Counter(clean_tokens(pos_text)).most_common(topn)
        neg_words = Counter(clean_tokens(neg_text)).most_common(topn)
        banks[bank] = {"pos": pos_words, "neg": neg_words, "counts": len(g)}
    return banks

def write_report(banks_info, plots, out_path: Path):
    lines = []
    lines.append("="*80)
    lines.append("BANKS REVIEW INSIGHTS & RECOMMENDATIONS")
    lines.append("="*80)
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append("")
    for bank, d in banks_info.items():
        lines.append(f"BANK: {bank}  (reviews: {d['counts']})")
        lines.append("  Drivers (evidence - top positive keywords):")
        if d["pos"]:
            lines.append("   - " + ", ".join(f"{w} ({c})" for w,c in d["pos"]))
        else:
            lines.append("   - (insufficient positive evidence)")
        lines.append("  Pain points (evidence - top negative keywords):")
        if d["neg"]:
            lines.append("   - " + ", ".join(f"{w} ({c})" for w,c in d["neg"]))
        else:
            lines.append("   - (insufficient negative evidence)")
        lines.append("  Recommendations:")
        # Basic rule-based recommendations tied to keywords
        recs = []
        neg_keys = {w for w,_ in d["neg"]}
        if {"crash","crashes","crashing","freeze","slow","lag"} & neg_keys:
            recs.append("Improve stability: add crash telemetry and prioritize fixes for top crash paths.")
        if {"otp","login","password","authenticate","verification"} & neg_keys:
            recs.append("Harden auth flows and improve error messaging and retry/OTP resilience.")
        if {"slow","loading","speed"} & neg_keys:
            recs.append("Optimize performance (network payloads, caching) and add progress indicators for slow paths.")
        if not recs:
            recs = ["Monitor reviews + telemetry for targeted improvements; run small UX tests."]
        for r in recs[:3]:
            lines.append("   - " + r)
        lines.append("")
    lines.append("Saved plots:")
    for p in plots:
        lines.append(f" - {p}")
    report_path = out_path / "analysis_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote report: %s", report_path)
    return report_path

def plot_sentiment_trend(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df.get("date", df.get("review_date")), errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    grp = (df.groupby(["bank_code","month","sentiment"]).size().reset_index(name="count"))
    total = df.groupby(["bank_code","month"]).size().reset_index(name="total")
    merged = grp.merge(total, on=["bank_code","month"])
    merged["pct"] = merged["count"] / merged["total"] * 100
    f = PLOTS_DIR / "sentiment_trend_by_bank.png"
    plt.figure(figsize=(10,5))
    sns.lineplot(data=merged, x="month", y="pct", hue="sentiment", style="bank_code", markers=True)
    plt.title("Monthly Sentiment % by Bank")
    plt.tight_layout()
    plt.savefig(f, dpi=150)
    plt.clf()
    return f

def plot_rating_distribution(df: pd.DataFrame):
    f = PLOTS_DIR / "rating_distribution_by_bank.png"
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x="rating", hue="bank_code", palette="tab10")
    plt.title("Rating Distribution by Bank")
    plt.tight_layout()
    plt.savefig(f, dpi=150)
    plt.clf()
    return f

def plot_top_themes(banks_info):
    # create simple theme freq bar from aggregated keywords
    rows = []
    for bank, d in banks_info.items():
        for w,c in d["pos"]+d["neg"]:
            rows.append({"bank":bank,"word":w,"count":c})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    top = df.groupby("word")["count"].sum().nlargest(12).index.tolist()
    df_top = df[df["word"].isin(top)].pivot_table(index="word", columns="bank", values="count", fill_value=0)
    f = PLOTS_DIR / "top_keywords_by_bank.png"
    df_top.plot(kind="bar", figsize=(10,6))
    plt.title("Top Keywords (by mentions) across banks")
    plt.tight_layout()
    plt.savefig(f, dpi=150)
    plt.clf()
    return f

def main():
    if not IN.exists():
        raise SystemExit(f"Cleaned CSV not found: {IN}")
    df = pd.read_csv(IN, low_memory=False)
    # ensure content column exists
    if "content" not in df.columns:
        df["content"] = df.get("review_text", df.get("text", ""))
    df = compute_sentiment(df)
    cov = coverage_check(df)
    df.to_csv(OUT_DIR / "analysis_results.csv", index=False)
    banks_info = per_bank_drivers(df, topn=6)
    plots = []
    plots.append(plot_sentiment_trend(df))
    plots.append(plot_rating_distribution(df))
    plots.append(plot_top_themes(banks_info))
    # attempt wordclouds if available
    try:
        from wordcloud import WordCloud
        for bank, d in banks_info.items():
            txt = " ".join([w for w,_ in d["pos"]+d["neg"]])
            if not txt.strip():
                continue
            wc = WordCloud(width=800, height=400, background_color="white").generate(txt)
            p = PLOTS_DIR / f"wordcloud_{bank}.png"
            plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
            plt.savefig(p, dpi=150); plt.clf()
            plots.append(p)
    except Exception:
        logging.info("wordcloud not available; skipping wordclouds")
    report = write_report(banks_info, plots, OUT_DIR)
    logging.info("Task4 outputs: report=%s, plots=%d, analysis_csv=%s", report, len([p for p in plots if p]), OUT_DIR / "analysis_results.csv")

if __name__ == "__main__":
    main()

# New cell: Run Task4 analysis and display results (paste into notebook)
from pathlib import Path
from IPython.display import display, Image, Markdown
import pandas as pd
import textwrap

# Run the task4 script (ensures outputs exist)
import subprocess, sys
subprocess.run([sys.executable, "-u", "../src/task4_insights.py"], cwd=str(Path().resolve()/"notebooks"))

OUT_DIR = Path("../data/analysis").resolve()
PLOTS = list((OUT_DIR/"plots").glob("*.png"))
REPORT = OUT_DIR/"analysis_report.txt"
ANALYSIS_CSV = OUT_DIR/"analysis_results.csv"

# Display report head
if REPORT.exists():
    txt = REPORT.read_text(encoding="utf-8")
    display(Markdown("### Analysis Report (head)"))
    display(Markdown("```\n" + "\n".join(txt.splitlines()[:60]) + "\n```"))
else:
    display(Markdown("**No report found at** " + str(REPORT)))

# Display plots inline
display(Markdown("### Generated plots"))
for p in sorted(PLOTS):
    display(Markdown(f"**{p.name}**"))
    display(Image(str(p), width=900))

# Show per-bank example quotes from cleaned CSV (evidence)
clean_csv = Path("../data/cleaned_reviews.csv")
if clean_csv.exists():
    df = pd.read_csv(clean_csv, low_memory=False)
    df['content'] = df.get('content', df.get('review_text', '')).astype(str)
    display(Markdown("### Example evidence (sample quotes)"))
    for bank, g in df.groupby("bank_code"):
        sample_pos = g[g['content'].str.lower().str.contains('good|best|nice|easy|excellent', na=False)].head(2)['content'].tolist()
        sample_neg = g[g['content'].str.lower().str.contains('crash|slow|otp|fail|bad|worst', na=False)].head(2)['content'].tolist()
        display(Markdown(f"**{bank}** — positives: {sample_pos or ['(none)']}; negatives: {sample_neg or ['(none)']}"))
else:
    display(Markdown("Cleaned CSV not found: " + str(clean_csv)))