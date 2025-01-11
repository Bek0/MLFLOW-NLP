import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pipelines.data_prepare_pipeline import data_preprocess_pipeline
from pipelines.train_pipline import model_train_pipeline

def parse_arguments():
    """Parse command line parameters"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Model Parameters')
    
    # Model Registry parameters
    parser.add_argument('--model_name', type=str, default='sentimentAnalysis',
                    help='Name for the registered model')
    parser.add_argument('--model_stage', type=str, default='Staging',
                    help='Stage for the model (Development/Staging/Production)')
    
    # Other parameters remain the same
    parser.add_argument('--tfidf_max_features', type=int, default=5000)
    parser.add_argument('--tfidf_max_df', type=float, default=0.95)
    parser.add_argument('--tfidf_min_df', type=int, default=1)
    parser.add_argument('--chi2_k', type=int, default=800)
    parser.add_argument('--lr_c', type=float, default=10.0)
    parser.add_argument('--lr_solver', type=str, default='lbfgs')
    
    return parser.parse_args()

def analyze_important_features(pipeline, feature_names):
    """Analyze and display important words"""
    select_k_best = pipeline.named_steps['chi2']
    top_features_indices = select_k_best.get_support(indices=True)
    top_feature_names = [feature_names[i] for i in top_features_indices]
    
    logistic_regression = pipeline.named_steps['classifier']
    coefficients = logistic_regression.coef_[0]
    
    top_indices = np.argsort(coefficients)[::-1]
    top_features = [top_feature_names[i] for i in top_indices]
    top_weights = [coefficients[i] for i in top_indices]
    
    positive_words = [(word, weight) for word, weight in zip(top_features, top_weights) if weight > 0]
    negative_words = [(word, weight) for word, weight in zip(top_features, top_weights) if weight < 0 and "fuc" not in word]
    
    return sorted(positive_words, key=lambda x: x[1], reverse=True), sorted(negative_words, key=lambda x: x[1])

def save_and_register_model(model_name, model_stage, run_id, accuracy):
    """Save and register the model in MLflow Model Registry"""
    client = MlflowClient()
    
    # Register the model if it doesn't exist
    try:
        client.create_registered_model(model_name)
        print(f"Created new registered model: {model_name}")
    except Exception:
        print(f"Model {model_name} already exists")
    
    # Create a new model version
    model_version = mlflow.register_model(
        f"runs:/{run_id}/sentiment-analysis-model",
        model_name
    )
    
    # Add description and tags
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"Sentiment analysis model with accuracy: {accuracy:.4f}"
    )
    
    # Transition the model to the specified stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=model_stage
    )
    
    # Add metadata as tags
    client.set_model_version_tag(model_name, model_version.version, "accuracy", f"{accuracy:.4f}")
    client.set_model_version_tag(model_name, model_version.version, "training_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    
    return model_version

def log_metrics_and_artifacts(model, y_test, y_pred, feature_names):
    """Log metrics and results in MLflow"""
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log other metrics
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_dict(clf_report, "classification_report.json")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    mlflow.log_dict({"confusion_matrix": conf_matrix.tolist()}, "confusion_matrix.json")
    
    # Log important features
    positive_words, negative_words = analyze_important_features(model, feature_names)
    
    positive_words_dict = {word: float(weight) for word, weight in positive_words[:10]}
    mlflow.log_dict({"top_10_positive_words": positive_words_dict}, "positive_words.json")
    
    negative_words_dict = {word: float(weight) for word, weight in negative_words[:10]}
    mlflow.log_dict({"top_10_negative_words": negative_words_dict}, "negative_words.json")
    
    # Log the model
    mlflow.sklearn.log_model(
        model,
        "sentiment-analysis-model",
        registered_model_name=None  # We'll register it separately
    )
    
    return accuracy

def main():
    args = parse_arguments()
    
    # Set up MLflow tracking
    mlflow.set_experiment("Sentiment_Analysis_NLP")
    
    with mlflow.start_run(run_name="sentiment-analysis-run") as run:
        
        # Log parameters
        mlflow.log_params(vars(args))
        df = pd.read_csv('comments.csv').dropna().drop(columns=['Unnamed: 0'])
        
        # Create and data preprocess pipeline
        preprocess_pipeline = data_preprocess_pipeline()  # Instantiate the pipeline
        df = preprocess_pipeline.fit_transform(df)  # Apply transformation       
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['body'], df['sentiment_label'],
            stratify=df['sentiment_label'],
            random_state=25,
            test_size=0.1
        )
        
        # Create and train pipeline
        model = model_train_pipeline(args)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Log metrics and model
        feature_names = model.named_steps['features'].get_feature_names_out()
        accuracy = log_metrics_and_artifacts(model, y_test, y_pred, feature_names)
        
        # Register the model
        model_version = save_and_register_model(
            args.model_name,
            args.model_stage,
            run.info.run_id,
            accuracy
        )
        
        print("\nModel Training and Registration Complete:")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model Name: {args.model_name}")
        print(f"Model Version: {model_version.version}")
        print(f"Model Stage: {args.model_stage}")
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()