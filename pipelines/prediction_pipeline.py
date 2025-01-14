# prediction_pipeline.py
import pandas as pd
import mlflow
from pipelines.data_prepare_pipeline import data_preprocess_pipeline

def predict_sentiment(texts, model_uri="models:/sentimentAnalysis/1"):
    """
    Predict sentiment for a list of texts using the pipeline.
    """
    if isinstance(texts, str):
        texts = [texts]  # Ensure that `texts` is a list, even for a single string

    # Create DataFrame with the texts
    df = pd.DataFrame({'body': texts})

    # Load the model
    model = mlflow.sklearn.load_model(model_uri)

    # Create and fit preprocessing pipeline
    preprocess_pipeline = data_preprocess_pipeline(model)
    
    # Process the texts using transform (not fit_transform, since we aren't training)
    processed_texts = preprocess_pipeline.transform(df)
    
    # Make predictions
    predictions = model.predict(processed_texts['body'])
    probabilities = model.predict_proba(processed_texts['body'])

    # Construct results
    results = []
    for i, text in enumerate(texts):
        result = {
            'text': text,
            'sentiment': 'Positive' if predictions[i] == 1 else 'Negative',
            'negative_probability': float(probabilities[i][0]),
            'positive_probability': float(probabilities[i][1])
        }
        results.append(result)

    return results
