<<<<<<< HEAD
from textblob import TextBlob
import re
import pandas as pd
from pathlib import Path
import sys
=======
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(filepath):
    """Load reviews from a CSV file."""
    return pd.read_csv(filepath)
>>>>>>> ddc2153 (Initial project setup and web scraper implementation)

def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
<<<<<<< HEAD
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
=======
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    text = ' '.join(tokens)
>>>>>>> ddc2153 (Initial project setup and web scraper implementation)
    
    return text

def analyze_sentiment(text):
<<<<<<< HEAD
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
=======
    """Analyze sentiment using TextBlob."""
    analysis = TextBlob(str(text))
    # Classify sentiment
    if analysis.sentiment.polarity > 0.1:
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def preprocess_reviews(df):
    """Preprocess the reviews DataFrame."""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Handle missing values
    df['content'] = df['content'].fillna('')
    
    # Clean text
    print("Cleaning review text...")
    df['cleaned_text'] = df['content'].apply(clean_text)
    
    # Analyze sentiment
    print("Analyzing sentiment...")
    df['sentiment'] = df['content'].apply(analyze_sentiment)
    
    # Convert date to datetime if it's not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df

def save_processed_data(df, output_path):
    """Save processed data to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main(input_file, output_file='data/processed/processed_reviews.csv'):
    """Main function to preprocess the data."""
    print(f"Loading data from {input_file}...")
    df = load_data(input_file)
    
    print("Preprocessing data...")
    processed_df = preprocess_reviews(df)
    
    print("Saving processed data...")
    save_processed_data(processed_df, output_file)
    
    print("Preprocessing completed successfully!")
    return processed_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/processed_reviews.csv'
        main(input_file, output_file)
    else:
        print("Please provide an input file path.")
>>>>>>> ddc2153 (Initial project setup and web scraper implementation)
