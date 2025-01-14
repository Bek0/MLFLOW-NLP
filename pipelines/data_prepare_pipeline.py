# data_prepare_pipeline.py
import pandas as pd
import re
from html import unescape
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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
        X['sentiment_label'] = X['sentiment'].apply(self.classify_sentiment).apply(pd.Series)
        return X

    def classify_sentiment(self, value):
        if value >= self.threshold:
            return 1
        return 0

class SentimentPredictor(BaseEstimator, TransformerMixin):
    """Transformer for making sentiment predictions"""
    
    def __init__(self, model=None):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.model is None:
            raise ValueError("Model not initialized. Please provide a model during initialization.")
            
        if isinstance(X, pd.DataFrame):
            texts = X['body']
        else:
            texts = X
            
        # Get predictions and probabilities
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        
        # Create DataFrame with original text and predictions
        result_df = pd.DataFrame({
            'body': texts,
            'prediction': predictions,
            'negative_prob': probabilities[:, 0],
            'positive_prob': probabilities[:, 1]
        })
        
        return result_df

def data_preprocess_pipeline(model=None):
    """
    Data preprocessing pipeline (clean, tokenize, and sentiment classification) for training.
    The model parameter is optional. If no model is provided, a default sentiment classifier is used.
    """
    steps = [
        ('text_cleaner', TextCleaner()),  # Preprocessing step
        ('text_tokenizer', TextTokenizer())  # Tokenization step
    ]
    
    # Add the appropriate classifier step based on the model parameter
    if model:
        steps.append(('predictor', SentimentPredictor(model=model)))
    else:
        steps.append(('sentiment_classifier', SentimentClassifier(threshold=0)))  # Default classifier
    
    return Pipeline(steps)
