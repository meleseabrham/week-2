from google_play_scraper import app, Sort, reviews_all
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import json
from tqdm import tqdm

# Bank app details (app IDs from Google Play Store)
BANK_APPS = {
    'CBE': 'com.cbe.cbe_birr_eritr',
    'BOA': 'com.bankofabyssinia.boa',
    'Dashen': 'com.dashen.mobilebanking'
}

def scrape_reviews(app_id, app_name, lang='en', country='et', sleep_mul=1):
    """
    Scrape reviews for a given app ID.
    
    Args:
        app_id (str): The app ID from Google Play Store
        app_name (str): Name of the bank
        lang (str): Language code (default: 'en')
        country (str): Country code (default: 'et' for Ethiopia)
        sleep_mul (int): Multiplier for sleep time between requests
        
    Returns:
        pd.DataFrame: DataFrame containing the scraped reviews
    """
    print(f"Scraping reviews for {app_name}...")
    
    # Scrape reviews with pagination
    all_reviews = []
    continuation_token = None
    
    for _ in tqdm(range(20)):  # Adjust range based on number of reviews needed
        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=200,  # Max allowed by the API
                continuation_token=continuation_token
            )
            
            all_reviews.extend(result)
            
            if not continuation_token or len(result) < 200:
                break
                
            # Be nice to Google's servers
            time.sleep(2 * sleep_mul)
            
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Add app name and source
    df['bank'] = app_name
    df['source'] = 'Google Play Store'
    
    # Convert timestamp to date
    if 'at' in df.columns:
        df['date'] = pd.to_datetime(df['at']).dt.date
    
    # Keep only necessary columns
    columns_to_keep = ['content', 'score', 'date', 'bank', 'source']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    return df

def save_reviews(df, output_dir='data/raw'):
    """
    Save reviews to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame containing reviews
        output_dir (str): Directory to save the file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/reviews_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} reviews to {filename}")
    return filename

def main():
    all_reviews = []
    
    for bank_name, app_id in BANK_APPS.items():
        try:
            # Scrape reviews
            df = scrape_reviews(app_id, bank_name)
            all_reviews.append(df)
            print(f"Scraped {len(df)} reviews for {bank_name}")
            
            # Save individual bank reviews
            save_reviews(df, f'data/raw/{bank_name}')
            
            # Be nice to Google's servers
            time.sleep(5)
            
        except Exception as e:
            print(f"Error processing {bank_name}: {e}")
            continue
    
    # Combine all reviews
    if all_reviews:
        combined_df = pd.concat(all_reviews, ignore_index=True)
        save_reviews(combined_df, 'data/processed')
        print(f"Total reviews collected: {len(combined_df)}")
    else:
        print("No reviews were collected.")

if __name__ == "__main__":
    main()