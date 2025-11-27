# In database.py, update the first line to include DateTime:
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.engine import URL
import os
from dotenv import load_dotenv
# Add at the top of database.py
import pandas as pd
import sqlalchemy

# Load environment variables
load_dotenv()

# Database connection string for SQL Server
connection_string = f"mssql+pyodbc://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
class Bank(Base):
    """Bank model to store bank information."""
    __tablename__ = 'banks'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False)  # e.g., 'CBE'
    name = Column(String(100), nullable=False)  # Full bank name
    app_name = Column(String(100))
    app_id = Column(String(100))
    
    reviews = relationship("Review", back_populates="bank")

class Review(Base):
    """Review model to store app reviews."""
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    review_id = Column(String(100), unique=True, nullable=False)  # Google Play review ID
    content = Column(Text)  # review_text
    rating = Column(Integer)
    date = Column(DateTime)  # Changed from Date to DateTime for more precision
    user_name = Column(String(100))
    thumbs_up = Column(Integer, default=0)
    reply_content = Column(Text, nullable=True)
    source = Column(String(50))
    app_version = Column(String(50), nullable=True)  # reviewCreatedVersion
    bank_id = Column(Integer, ForeignKey('banks.id'))
    
    # Additional fields for analysis
    sentiment = Column(String(20), nullable=True)
    cleaned_text = Column(Text, nullable=True)
    
    bank = relationship("Bank", back_populates="reviews")

def init_db():
    """Initialize the database by dropping existing tables and creating new ones."""
    # Drop all existing tables
    Base.metadata.drop_all(engine)
    # Create all tables
    Base.metadata.create_all(engine)
    print("Database reinitialized successfully with updated schema!")

def analyze_sentiment(text):
    """Analyze sentiment of the text using TextBlob."""
    from textblob import TextBlob
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

def clean_text(text):
    """Clean and preprocess text data."""
    import re
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

def get_unprocessed_reviews_count():
    """Get count of reviews that need sentiment analysis."""
    session = Session()
    try:
        return session.query(Review).filter(
            (Review.sentiment == None) | 
            (Review.cleaned_text == None)
        ).count()
    finally:
        session.close()

def update_reviews_sentiment(batch_size=100):
    """Update reviews with sentiment analysis and cleaned text.
    
    Args:
        batch_size (int): Number of reviews to process in one batch
    
    Returns:
        int: Number of reviews updated
    """
    session = Session()
    updated_count = 0
    
    try:
        # Get unprocessed reviews
        reviews_to_update = session.query(Review).filter(
            (Review.sentiment == None) | 
            (Review.cleaned_text == None)
        ).limit(batch_size).all()
        
        for review in reviews_to_update:
            # Clean text and analyze sentiment if not already done
            if not review.cleaned_text and review.content:
                review.cleaned_text = clean_text(review.content)
            
            if not review.sentiment and review.content:
                review.sentiment = analyze_sentiment(review.content)
            
            updated_count += 1
        
        if updated_count > 0:
            session.commit()
            print(f"Updated {updated_count} reviews with sentiment analysis")
        
        return updated_count
        
    except Exception as e:
        session.rollback()
        print(f"Error updating reviews: {e}")
        raise
    finally:
        session.close()

def save_reviews_to_database(reviews_data, bank_code, bank_name, app_id):
    """Save reviews to the database with sentiment analysis.
    
    Args:
        reviews_data (list): List of review dictionaries
        bank_code (str): Bank code (e.g., 'CBE')
        bank_name (str): Full bank name
        app_id (str): Application ID from Google Play
    
    Returns:
        tuple: (new_reviews_count, updated_reviews_count)
    """
    session = Session()
    new_reviews = 0
    updated_reviews = 0
    
    try:
        # Check if bank exists, if not create it
        bank = session.query(Bank).filter_by(code=bank_code).first()
        if not bank:
            bank = Bank(
                code=bank_code,
                name=bank_name,
                app_name=f"{bank_name} Mobile",
                app_id=app_id
            )
            session.add(bank)
            session.commit()
        
        # Process reviews
        for review_data in reviews_data:
            # Check if review exists
            existing = session.query(Review).filter_by(review_id=review_data.get('review_id', '')).first()
            
            if existing:
                # Update existing review if needed
                update_needed = False
                
                # Check if we need to update any fields
                if not existing.cleaned_text and review_data.get('review_text'):
                    existing.cleaned_text = clean_text(review_data['review_text'])
                    update_needed = True
                
                if not existing.sentiment and review_data.get('review_text'):
                    existing.sentiment = analyze_sentiment(review_data['review_text'])
                    update_needed = True
                
                if update_needed:
                    updated_reviews += 1
                
                continue
            
            # Add new review
            cleaned_text = clean_text(review_data.get('review_text', ''))
            sentiment = analyze_sentiment(review_data.get('review_text', ''))
            
            new_review = Review(
                review_id=review_data.get('review_id', ''),
                content=review_data.get('review_text', ''),
                cleaned_text=cleaned_text,
                sentiment=sentiment,
                rating=review_data.get('rating', 0),
                date=review_data.get('review_date'),
                user_name=review_data.get('user_name', 'Anonymous'),
                thumbs_up=review_data.get('thumbs_up', 0),
                reply_content=review_data.get('reply_content', ''),
                source=review_data.get('source', 'Google Play'),
                app_version=review_data.get('app_id', ''),
                bank_id=bank.id
            )
            
            session.add(new_review)
            new_reviews += 1
        
        # Commit all changes
        session.commit()
        
        if new_reviews > 0 or updated_reviews > 0:
            print(f"Saved {new_reviews} new reviews and updated {updated_reviews} existing reviews for {bank_name}")
        else:
            print("No new or updated reviews to save.")
        
        return new_reviews, updated_reviews
        
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        raise
    finally:
        session.close()

def load_from_database(bank_name=None, include_sentiment=True):
    """Load reviews from the database.
    
    Args:
        bank_name (str, optional): Filter by bank name
        include_sentiment (bool): Whether to include sentiment analysis in the results
        
    Returns:
        pd.DataFrame: DataFrame containing reviews and sentiment analysis
    """
    session = Session()
    try:
        query = session.query(Review)
        
        if bank_name:
            query = query.join(Bank).filter(Bank.name == bank_name)
        
        # If sentiment analysis is requested but not all reviews have it, process them
        if include_sentiment:
            unprocessed_count = get_unprocessed_reviews_count()
            if unprocessed_count > 0:
                print(f"Found {unprocessed_count} reviews without sentiment analysis. Processing...")
                update_reviews_sentiment(batch_size=unprocessed_count)
        
        # Load the data into a DataFrame
        df = pd.read_sql(query.statement, session.bind)
        
        # Add bank info to the DataFrame
        if 'bank_id' in df.columns:
            banks = pd.read_sql(session.query(Bank).statement, session.bind)
            df = df.merge(banks[['id', 'name', 'code']], 
                         left_on='bank_id', 
                         right_on='id', 
                         suffixes=('', '_bank'))
        
        return df
        
    except Exception as e:
        print(f"Error loading from database: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def export_to_csv(filename='data/processed/bank_reviews.csv', bank_name=None):
    """Export reviews to a CSV file with sentiment analysis.
    
    Args:
        filename (str): Path to the output CSV file
        bank_name (str, optional): Filter by bank name
    """
    try:
        # Ensure the output directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Load data with sentiment analysis
        df = load_from_database(bank_name=bank_name, include_sentiment=True)
        
        if df.empty:
            print("No data to export.")
            return False
        
        # Select and rename columns for export
        columns = {
            'review_id': 'review_id',
            'content': 'review_text',
            'cleaned_text': 'cleaned_text',
            'sentiment': 'sentiment',
            'rating': 'rating',
            'date': 'review_date',
            'user_name': 'user_name',
            'thumbs_up': 'thumbs_up',
            'reply_content': 'reply_content',
            'source': 'source',
            'app_version': 'app_version',
            'name': 'bank_name',
            'code': 'bank_code'
        }
        
        # Filter and rename columns
        df = df[list(columns.keys())].rename(columns=columns)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Successfully exported {len(df)} reviews to {filename}")
        return True
        
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management for bank reviews')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Init DB command
    init_parser = subparsers.add_parser('init', help='Initialize the database')
    
    # Update sentiment command
    update_parser = subparsers.add_parser('update', help='Update sentiment analysis')
    update_parser.add_argument('--batch-size', type=int, default=100, 
                             help='Number of reviews to process in one batch')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export reviews to CSV')
    export_parser.add_argument('--output', '-o', default='data/processed/bank_reviews.csv',
                             help='Output CSV file path')
    export_parser.add_argument('--bank', help='Filter by bank name')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init_db()
    elif args.command == 'update':
        updated = update_reviews_sentiment(batch_size=args.batch_size)
        print(f"Updated {updated} reviews with sentiment analysis")
    elif args.command == 'export':
        export_to_csv(filename=args.output, bank_name=args.bank)
    elif args.command == 'stats':
        session = Session()
        try:
            # Get total reviews count
            total_reviews = session.query(Review).count()
            print(f"Total reviews: {total_reviews}")
            
            # Get reviews by bank
            print("\nReviews by bank:")
            bank_counts = session.query(Bank.name, Bank.code, sqlalchemy.func.count(Review.id))\
                .outerjoin(Review, Bank.id == Review.bank_id)\
                .group_by(Bank.name, Bank.code)\
                .all()
            for bank_name, bank_code, count in bank_counts:
                print(f"  {bank_name} ({bank_code}): {count} reviews")
            
            # Get sentiment distribution
            print("\nSentiment distribution:")
            sentiment_counts = session.query(
                Review.sentiment, 
                sqlalchemy.func.count(Review.id)
            ).group_by(Review.sentiment).all()
            
            for sentiment, count in sentiment_counts:
                if sentiment:
                    print(f"  {sentiment.capitalize()}: {count} ({(count/total_reviews*100):.1f}%)")
                else:
                    print(f"  Not analyzed: {count} ({(count/total_reviews*100):.1f}%)")
            
            # Get rating distribution
            print("\nRating distribution:")
            rating_counts = session.query(
                Review.rating, 
                sqlalchemy.func.count(Review.id)
            ).group_by(Review.rating).order_by(Review.rating.desc()).all()
            
            for rating, count in rating_counts:
                if rating:
                    print(f"  {int(rating)} stars: {count} ({(count/total_reviews*100):.1f}%)")
                else:
                    print(f"  No rating: {count} ({(count/total_reviews*100):.1f}%)")
                    
        finally:
            session.close()
    else:
        parser.print_help()