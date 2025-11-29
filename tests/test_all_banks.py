import sys
import os
import time  # Add this import at the top of the file
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import init_db, save_reviews_to_database
from src.scraper import PlayStoreScraper
from src.config import BANK_NAMES, APP_IDS

def test_all_bank_reviews():
    # Initialize database
    init_db()
    
    # Initialize scraper
    scraper = PlayStoreScraper()
    
    total_saved = 0
    
    # Process each bank
    for bank_code, app_id in APP_IDS.items():
        print(f"\n{'='*50}")
        print(f"Processing {BANK_NAMES[bank_code]} ({bank_code})")
        print(f"App ID: {app_id}")
        print("="*50)
        
        # Scrape reviews
        print("Scraping reviews...")
        reviews_data = scraper.scrape_reviews(app_id, count=400)
        
        if not reviews_data:
            print(f"Warning: No reviews found for {bank_code}")
            continue
            
        # Process the reviews
        processed_reviews = scraper.process_reviews(reviews_data, bank_code)
        print(f"Processed {len(processed_reviews)} reviews")
        
        # Save to database
        count = save_reviews_to_database(
            reviews_data=processed_reviews,
            bank_code=bank_code,
            bank_name=BANK_NAMES[bank_code],
            app_id=app_id
        )
        
        print(f"Saved {count} new reviews to database")
        total_saved += count
        
        # Small delay between banks
        print("Waiting 5 seconds before next bank...")
        time.sleep(5)
    
    print(f"\n{'='*50}")
    print(f"Completed! Total reviews saved: {total_saved}")
    print("="*50)

if __name__ == "__main__":
    test_all_bank_reviews()