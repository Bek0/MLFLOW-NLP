from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from html import unescape
import string
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing class to include text cleaning and lemmatization
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def fit(self, X, y=None):
        return self  # No fitting needed for text transformation
    
    def transform(self, X):
        # Apply preprocessing to each text sample in X
        return [self.preprocess_text(text) for text in X]
    
    def preprocess_text(self, text):
        """Preprocess text using common NLP techniques"""
        # Clean text
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
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)

# Create a pipeline for text processing and prediction
def model_prediction_pipeline(model):
    """Creates a pipeline for text preprocessing for model prediction."""
    return Pipeline([
        ('preprocessor', TextPreprocessor()),  # Preprocessing step
        ('classifier', model)  # Model prediction step
    ])