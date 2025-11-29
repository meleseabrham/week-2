# Bank App Review Analysis

This project analyzes customer reviews of Ethiopian bank mobile applications to extract actionable insights for improving user experience and satisfaction.

## 📋 Project Overview

The project consists of two main phases:
1. **Data Collection & Preprocessing**: Scraping and cleaning bank app reviews
2. **Sentiment & Thematic Analysis**: Analyzing reviews to identify key themes and sentiment trends

## 🚀 Features

### Task 1: Data Collection & Preprocessing
- Scraped 6,400+ reviews from multiple Ethiopian bank apps
- Implemented robust error handling and rate limiting
- Cleaned and preprocessed data for analysis
- Stored results in structured CSV format

### Task 2: Sentiment & Thematic Analysis
- Performed sentiment analysis using TextBlob
- Identified 5 key themes across all bank apps
- Generated comprehensive analysis reports
- Visualized key findings

## 📊 Key Findings

### Sentiment Distribution
- Positive: 52.8%
- Neutral: 38.1%
- Negative: 9.1%

### Top Themes Identified
1. **User Interface** (34% of reviews)
   - Navigation difficulties
   - Design improvements
2. **Performance** (28%)
   - App crashes
   - Slow loading times
3. **Features** (22%)
   - Payment options
   - Search functionality
4. **Security** (10%)
   - Login issues
   - Authentication
5. **Customer Support** (6%)
   - Response times
   - Issue resolution

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/meleseabrham/week-2.git](https://github.com/meleseabrham/week-2.git)
   cd week-2
   python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
week-2/
├── data/
│   ├── raw/               # Raw scraped data
│   └── analysis/          # Analysis results and reports
├── src/
│   ├── scraper.py         # Web scraping logic
│   ├── preprocess.py      # Data cleaning and preprocessing
│   ├── analysis.py        # Sentiment and thematic analysis
│   └── database.py        # Database operations
├── .gitignore
├── requirements.txt
└── README.md
python -m src.scraper
python -m src.analysis