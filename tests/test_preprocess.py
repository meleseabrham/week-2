import unittest
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_text, analyze_sentiment

class TestPreprocessing(unittest.TestCase):
    def test_clean_text(self):
        # Test URL removal
        self.assertEqual(clean_text("Check this https://example.com"), "check example")
        
        # Test HTML tag removal
        self.assertEqual(clean_text("<p>Hello</p>"), "hello")
        
        # Test punctuation removal
        self.assertEqual(clean_text("Hello, world!"), "hello world")
        
        # Test number removal
        self.assertEqual(clean_text("Version 2.0 is here"), "version is here")
    
    def test_analyze_sentiment(self):
        # Test positive sentiment
        self.assertEqual(analyze_sentiment("I love this app!"), "positive")
        
        # Test negative sentiment
        self.assertEqual(analyze_sentiment("I hate this app!"), "negative")
        
        # Test neutral sentiment
        self.assertEqual(analyze_sentiment("This is an app."), "neutral")

if __name__ == "__main__":
    unittest.main()