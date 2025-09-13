# utils/sentiment_analysis.py
from textblob import TextBlob
import re

def clean_tweet(tweet):
    """
    Clean tweet text by removing links, special characters, etc.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analyze_sentiment(tweet):
    """
    Analyze sentiment of a tweet using TextBlob
    """
    analysis = TextBlob(clean_tweet(tweet))
    return analysis.sentiment.polarity