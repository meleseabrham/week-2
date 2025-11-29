from textblob import TextBlob
import re
import pandas as pd
from pathlib import Path
import sys

def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def analyze_sentiment(text):
    """Analyze sentiment of the text using TextBlob."""
    try:
        analysis = TextBlob(str(text))
        # Classify sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
    except:
        return 'neutral'

def preprocess_reviews(df):
    """Preprocess reviews DataFrame."""
    # Clean text
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Analyze sentiment
    df['sentiment'] = df['review_text'].apply(analyze_sentiment)
    
    return df

def update_database_sentiments():
    """Update database with sentiment analysis and cleaned text."""
    from src.database import Session, Review
    
    session = Session()
    
    try:
        # Get all reviews that need processing
        reviews = session.query(Review).filter(
            (Review.sentiment == None) | 
            (Review.cleaned_text == None)
        ).all()
        
        print(f"Found {len(reviews)} reviews to process...")
        
        for i, review in enumerate(reviews, 1):
            # Clean text
            review.cleaned_text = clean_text(review.content)
            
            # Analyze sentiment
            review.sentiment = analyze_sentiment(review.content)
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i}/{len(reviews)} reviews")
                session.commit()  # Commit every 100 records
        
        session.commit()
        print("Successfully updated all reviews with sentiment analysis!")
        
    except Exception as e:
        session.rollback()
        print(f"Error updating reviews: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    update_database_sentiments()