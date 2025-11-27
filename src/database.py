from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection string
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fintech_reviews')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
class Bank(Base):
    """Bank model to store bank information."""
    __tablename__ = 'banks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    app_name = Column(String(100))
    app_id = Column(String(100))
    
    reviews = relationship("Review", back_populates="bank")

class Review(Base):
    """Review model to store app reviews."""
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    rating = Column(Integer)
    date = Column(Date)
    sentiment = Column(String(20))
    cleaned_text = Column(Text)
    source = Column(String(50))
    bank_id = Column(Integer, ForeignKey('banks.id'))
    
    bank = relationship("Bank", back_populates="reviews")

def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")

def save_to_database(df, bank_name):
    """Save reviews to the database."""
    # Create a new session
    session = Session()
    
    try:
        # Check if bank exists, if not create it
        bank = session.query(Bank).filter_by(name=bank_name).first()
        if not bank:
            bank = Bank(
                name=bank_name,
                app_name=f"{bank_name.capitalize()} Mobile",
                app_id=f"com.{bank_name.lower()}.mobile"
            )
            session.add(bank)
            session.commit()
        
        # Prepare reviews data
        reviews_data = []
        for _, row in df.iterrows():
            review = Review(
                content=row.get('content', ''),
                rating=row.get('score', row.get('rating', 0)),
                date=row.get('date'),
                sentiment=row.get('sentiment', 'neutral'),
                cleaned_text=row.get('cleaned_text', ''),
                source=row.get('source', 'Google Play Store'),
                bank_id=bank.id
            )
            reviews_data.append(review)
        
        # Add all reviews
        session.bulk_save_objects(reviews_data)
        session.commit()
        print(f"Successfully saved {len(reviews_data)} reviews for {bank_name} to the database.")
        
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        raise
    finally:
        session.close()

def load_from_database(bank_name=None):
    """Load reviews from the database."""
    session = Session()
    try:
        query = session.query(Review)
        if bank_name:
            query = query.join(Bank).filter(Bank.name == bank_name)
        
        df = pd.read_sql(query.statement, session.bind)
        return df
    except Exception as e:
        print(f"Error loading from database: {e}")
        return pd.DataFrame()
    finally:
        session.close()

if __name__ == "__main__":
    # Initialize the database
    init_db()