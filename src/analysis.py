import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
from collections import defaultdict

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Analyze sentiment using TextBlob with neutral threshold"""
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    return 'neutral', polarity

class ThemeAnalyzer:
    def __init__(self):
        self.theme_keywords = {
            'User Interface': ['app', 'ui', 'ux', 'design', 'interface', 'screen', 'button', 'layout'],
            'Performance': ['slow', 'fast', 'speed', 'lag', 'crash', 'freeze', 'loading', 'response'],
            'Features': ['transfer', 'payment', 'bill', 'notification', 'transaction', 'balance', 'account'],
            'Security': ['login', 'password', 'secure', 'otp', 'verification', 'authenticate'],
            'Customer Support': ['support', 'help', 'service', 'response', 'contact', 'assistance']
        }
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=1000
        )

    def extract_themes(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extract and categorize themes from reviews"""
        X = self.vectorizer.fit_transform(texts)
        feature_array = self.vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(X.mean(axis=0)).ravel()
        
        top_indices = np.argsort(tfidf_scores)[::-1][:50]
        keywords_with_scores = [
            (feature_array[i], tfidf_scores[i]) 
            for i in top_indices 
            if tfidf_scores[i] > 0.01
        ]
        
        theme_matches = defaultdict(list)
        for keyword, score in keywords_with_scores:
            for theme, keywords in self.theme_keywords.items():
                if any(kw in keyword for kw in keywords):
                    theme_matches[theme].append((keyword, score))
        
        theme_scores = {
            theme: sum(score for _, score in matches)
            for theme, matches in theme_matches.items()
        }
        top_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            theme: [kw for kw, _ in sorted(matches, key=lambda x: x[1], reverse=True)[:5]]
            for theme, matches in theme_matches.items()
            if theme in dict(top_themes)
        }

def analyze_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Main analysis pipeline"""
    analyzer = ThemeAnalyzer()
    results = []
    
    for bank_name, group in tqdm(df.groupby('name'), desc="Analyzing banks"):
        print(f"\nAnalyzing {bank_name}...")
        
        # Sentiment Analysis
        sentiments = [analyze_sentiment(text) for text in group['content']]
        group['sentiment'], group['sentiment_score'] = zip(*sentiments)
        
        # Theme Analysis
        themes = analyzer.extract_themes(group['content'].tolist())
        
        # Assign themes to each review
        for theme, keywords in themes.items():
            group[f"theme_{theme.lower().replace(' ', '_')}"] = group['content'].apply(
                lambda x: any(keyword in str(x).lower() for keyword in keywords)
            )
        
        results.append(group)
    
    return pd.concat(results)

def generate_reports(df: pd.DataFrame, output_dir: str = 'data/analysis'):
    """Generate analysis reports"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save full results
    output_path = os.path.join(output_dir, 'analysis_results.csv')
    df.to_csv(output_path, index=False)
    
    # 2. Generate summary report
    report = []
    report.append("="*60)
    report.append("BANK REVIEW ANALYSIS REPORT")
    report.append("="*60)
    
    # Overall sentiment
    sentiment_dist = df['sentiment'].value_counts(normalize=True).mul(100)
    report.append("\nOVERALL SENTIMENT DISTRIBUTION:")
    for sentiment, pct in sentiment_dist.items():
        report.append(f"- {sentiment.capitalize()}: {pct:.1f}%")
    
    # Analysis by bank
    for bank_name, group in df.groupby('name'):
        report.append("\n" + "="*60)
        report.append(f"BANK: {bank_name.upper()}")
        report.append("="*60)
        
        # Sentiment by bank
        bank_sentiment = group['sentiment'].value_counts(normalize=True).mul(100)
        report.append("\nSENTIMENT:")
        for sentiment, pct in bank_sentiment.items():
            report.append(f"- {sentiment.capitalize()}: {pct:.1f}%")
        
        # Themes
        theme_columns = [col for col in df.columns if col.startswith('theme_')]
        theme_counts = group[theme_columns].sum().sort_values(ascending=False)
        
        report.append("\nTOP THEMES:")
        for theme, count in theme_counts.items():
            if count > 0:
                theme_name = theme.replace('theme_', '').replace('_', ' ').title()
                report.append(f"- {theme_name}: {int(count)} mentions")
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return output_path, report_path

if __name__ == "__main__":
    from .database import load_from_database
    
    print("Starting bank review analysis...")
    
    # Load data
    print("Loading data from database...")
    df = load_from_database()
    
    # Run analysis
    print("\nAnalyzing reviews...")
    results = analyze_reviews(df)
    
    # Generate reports
    print("\nGenerating reports...")
    results_path, report_path = generate_reports(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_path}")
    print(f"Report saved to: {report_path}")
    
    # Print summary
    with open(report_path, 'r') as f:
        print("\n" + "="*60)
        print("REPORT SUMMARY")
        print("="*60)
        print('\n'.join(f.readlines()[:20]))  # Print first 20 lines
        print("\n... (see full report for details)")