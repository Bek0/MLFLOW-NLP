# Sentiment Analysis Project

## Overview
This project is a comprehensive pipeline for training, testing, and deploying a sentiment analysis model using Natural Language Processing (NLP) techniques. The project integrates several stages, from data preprocessing to model training, evaluation, and deployment using **MLflow** for model tracking and management.

---

## Directory Structure
```plaintext
.
├── README.md
├── main.py
├── pipelines/
│   ├── data_prepare_pipeline.py
│   ├── train_pipeline.py
│   └── prediction_pipeline.py
├── comments.csv
└── requirements.txt
```

## Files and Components

### `main.py`
The main entry point of the project. This script orchestrates the entire workflow, from preprocessing the dataset to training the model, logging metrics, and deploying the model. 

#### Responsibilities:
1. **Argument Parsing**:
   - Parses command-line arguments for key model parameters such as:
     - `tfidf_max_features`: The maximum number of features for the TF-IDF vectorizer.
     - `chi2_k`: Number of features to retain after Chi-Square feature selection.
     - Other hyperparameters for the Logistic Regression model.
   - Ensures flexibility in experimenting with different configurations.

2. **Dataset Loading and Preprocessing**:
   - Uses the `data_preprocess_pipeline` to clean and tokenize the dataset.

3. **Model Training**:
   - Trains the sentiment analysis model using the `model_train_pipeline`.
   - Evaluates the model on training and validation datasets.

4. **MLflow Integration**:
   - Logs key metrics such as:
     - Accuracy
     - Precision, Recall, F1-Score
     - Confusion Matrix and Classification Report
   - Logs artifacts including:
     - Feature importance visualizations
     - Trained model files
   - Registers the trained model in the MLflow Model Registry.

5. **Feature Analysis**:
   - Extracts and logs the most important words (positive and negative) that the model uses to make predictions.

#### Key Functions:
- **`parse_arguments()`**:
  - Dynamically parses user-provided configurations from the command line.
  - Ensures compatibility for hyperparameter tuning experiments.

- **`analyze_important_features()`**:
  - Identifies the most influential features (words) in the model’s decision-making process.
  - Provides insights into positive and negative sentiment words.

- **`save_and_register_model()`**:
  - Saves the trained model in a format compatible with MLflow.
  - Registers the model in MLflow’s Model Registry, enabling deployment to production.

- **`log_metrics_and_artifacts()`**:
  - Logs all relevant metrics, artifacts, and visualizations to MLflow for reproducibility and experiment tracking.

---

### `pipelines/`
This folder contains modular scripts that handle specific stages of the sentiment analysis pipeline. Each pipeline is designed to be reusable and modular, enabling seamless integration and debugging.

#### `data_prepare_pipeline.py`
This script is responsible for preparing the raw dataset for model training. It includes comprehensive text preprocessing steps.

##### Components:
1. **`TextCleaner`**:
   - Cleans raw text data by performing the following operations:
     - Removing HTML tags, URLs, and special characters.
     - Lowercasing the text and removing punctuation.
     - Stripping extra whitespace to standardize input format.

2. **`TextTokenizer`**:
   - Tokenizes text into individual words or tokens.
   - Removes common stopwords to focus on meaningful words.
   - Lemmatizes tokens using **NLTK** tools to reduce words to their base forms (e.g., "running" → "run").

3. **`SentimentClassifier`**:
   - Maps text data to sentiment labels:
     - Positive sentiment: Numerical score ≥ 0.
     - Negative sentiment: Numerical score < 0.

##### Outputs:
- Cleaned and tokenized text ready for feature extraction.
- Labeled data for supervised learning.

---

#### `train_pipeline.py`
Defines the training process, integrating feature extraction, selection, and classification into a cohesive workflow.

##### Components:
1. **`TfidfVectorizer`**:
   - Converts textual data into numerical format using Term Frequency-Inverse Document Frequency (TF-IDF).
   - Configurable to limit the maximum number of features using `max_features`.

2. **`SelectKBest`**:
   - Performs feature selection based on statistical significance.
   - Retains the top `k` features using the Chi-Square test.

3. **`LogisticRegression`**:
   - A robust and interpretable model for binary classification tasks.
   - Configurable through hyperparameters such as regularization strength and solver.

##### Workflow:
1. Preprocesses text data using `data_preprocess_pipeline`.
2. Extracts features with TF-IDF.
3. Selects the most significant features.
4. Trains the Logistic Regression model on the transformed data.

##### Outputs:
- A trained Logistic Regression model.
- Key metrics such as training accuracy, validation accuracy, and confusion matrix.

---

#### `prediction_pipeline.py`
Handles real-time predictions, enabling the trained model to classify unseen text data.

##### Responsibilities:
1. **Model Loading**:
   - Fetches the trained model from MLflow’s Model Registry.

2. **Prediction**:
   - Processes incoming text data using the preprocessing pipeline.
   - Outputs predictions along with probabilities for each class.

##### Features:
- Supports batch predictions for multiple inputs.
- Outputs predictions in a user-friendly format:
  - Predicted Class (Positive/Negative)
  - Confidence Scores (Probabilities for each class)

##### Outputs:
- Predictions with confidence scores, making the pipeline suitable for real-world deployment.

---

#### `load_model.ipynb`
A Jupyter Notebook designed for testing and exploring predictions with the trained sentiment analysis model.

##### Workflow:
1. **Model Loading**:
   - Uses MLflow to load the latest version of the trained model.
   - Example model URI: `"models:/sentimentAnalysis@challenger"`.
   - The model is fetched directly from the MLflow Model Registry.

2. **Pipeline Integration**:
   - Creates a prediction pipeline by integrating the loaded model with `model_prediction_pipeline`.

3. **Prediction Functionality**:
   - **`predict_sentiment(text)`**:
     - Accepts a text input and returns:
       - Predicted sentiment (`Positive` or `Negative`).
       - Associated probabilities for each class (`negative_probability` and `positive_probability`).

4. **Batch Predictions**:
   - Demonstrates how to process a list of texts for sentiment predictions in a loop.
   - Outputs predictions and associated probabilities in a structured format.


### Additional Notes
- The project integrates **MLflow** for experiment tracking, model versioning, and deployment readiness.
- All pipelines are designed to be modular, making it easier to adapt and extend the project to new datasets or additional tasks.
- The clear separation of concerns between preprocessing, training, and prediction ensures maintainability and scalability.
