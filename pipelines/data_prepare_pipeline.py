# data_pipeline.py
import pandas as pd
import re
from html import unescape
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TextCleaner(BaseEstimator, TransformerMixin):
    """Transformer for cleaning text data"""
    def transform(self, X):
        X['body'] = X['body'].apply(self.cleaning_text)
        return X
    def fit(self, X, y=None):
        # Implement any fitting logic here if needed
        # If no fitting is needed, just return self
        return self
    
    def cleaning_text(self, text):
        text = unescape(text)
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
        text = re.sub(r'\b\w*\d\w*\b', '', text)
        return text

class TextTokenizer(BaseEstimator, TransformerMixin):
    """Transformer for tokenizing and processing text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    def fit(self, X, y=None):
        # Implement any fitting logic here if needed
        # If no fitting is needed, just return self
        return self
    
    def transform(self, X):
        X['body'] = X['body'].apply(self.tokenize_and_process)
        return X

    def tokenize_and_process(self, text):
        """Tokenize the text, remove stopwords, and lemmatize"""
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

class SentimentClassifier(BaseEstimator, TransformerMixin):
    """Transformer for classifying sentiment"""
    
    def __init__(self, threshold=0):
        self.threshold = threshold
    def fit(self, X, y=None):
        # Implement any fitting logic here if needed
        # If no fitting is needed, just return self
        return self
    
    def transform(self, X):
        X[['sentiment_class', 'sentiment_label']] = X['sentiment'].apply(self.classify_sentiment).apply(pd.Series)
        return X

    def classify_sentiment(self, value):
        if value >= self.threshold:
            return 'Positive', 1
        return 'Negative', 0

def data_preprocess_pipeline():
    """Data preprocessing pipeline (clean, tokenize and sentiment_classifier) for training"""
    return Pipeline([
        ('text_cleaner', TextCleaner()),
        ('text_tokenizer', TextTokenizer()),
        ('sentiment_classifier', SentimentClassifier(threshold=0))
    ])