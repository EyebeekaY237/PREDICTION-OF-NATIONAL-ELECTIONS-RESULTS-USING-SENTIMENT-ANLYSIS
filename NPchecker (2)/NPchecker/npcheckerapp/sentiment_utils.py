# sentiment_utils.py - UPDATED
import joblib
import re
import os
import numpy as np
from django.conf import settings

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Try to load ML models, fallback to rule-based if not available"""
        try:
            model_path = os.path.join(settings.BASE_DIR, 'sentiment_model.pkl')
            vectorizer_path = os.path.join(settings.BASE_DIR, 'tfidf_vectorizer.pkl')
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("ML sentiment models loaded successfully")
            else:
                print("ML model files not found, using rule-based analysis")
                
        except Exception as e:
            print(f"Error loading ML models: {e}, using rule-based analysis")
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s\']', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with ML model or fallback"""
        cleaned_text = self.clean_text(text)
        
        if self.model and self.vectorizer:
            try:
                text_tfidf = self.vectorizer.transform([cleaned_text])
                prediction = self.model.predict(text_tfidf)[0]
                
                # Convert categorical prediction to numerical score
                sentiment_map = {
                    'strong_negative': -0.9,
                    'negative': -0.6,
                    'neutral': 0.0,
                    'positive': 0.6,
                    'strong_positive': 0.9
                }
                return sentiment_map.get(prediction, 0.0)
                
            except Exception as e:
                print(f"ML prediction failed: {e}, using rule-based")
                return self._rule_based_sentiment(cleaned_text)
        else:
            return self._rule_based_sentiment(cleaned_text)
    
    def _rule_based_sentiment(self, text):
        """Enhanced rule-based sentiment analysis"""
        text = text.lower()
        
        # More comprehensive sentiment dictionaries
        strong_positive = ['excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
                          'brilliant', 'superb', 'perfect', 'exceptional', 'phenomenal']
        positive = ['good', 'great', 'nice', 'better', 'improved', 'positive',
                   'support', 'like', 'love', 'approve', 'recommend']
        negative = ['bad', 'poor', 'worse', 'negative', 'dislike', 'hate',
                   'problem', 'issue', 'concern', 'criticize', 'oppose']
        strong_negative = ['terrible', 'awful', 'horrible', 'disastrous', 'catastrophic',
                          'dreadful', 'appalling', 'atrocious', 'deplorable', 'abysmal']
        
        strong_pos_count = sum(1 for word in strong_positive if word in text)
        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)
        strong_neg_count = sum(1 for word in strong_negative if word in text)
        
        total_positive = strong_pos_count * 2 + pos_count
        total_negative = strong_neg_count * 2 + neg_count
        
        if total_positive > total_negative:
            if strong_pos_count > 0:
                return 0.8
            else:
                return 0.5
        elif total_negative > total_positive:
            if strong_neg_count > 0:
                return -0.8
            else:
                return -0.5
        else:
            return 0.0
    
    def batch_analyze(self, texts):
        """Analyze multiple texts at once"""
        return [self.analyze_sentiment(text) for text in texts]

# Create singleton instance
sentiment_analyzer = EnhancedSentimentAnalyzer()