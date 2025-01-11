# train_pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

def model_train_pipeline(args):
    """
    pipeline for feature extraction, feature selection, and classification model.
    """
    tfidf = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        sublinear_tf=False,
        max_df=args.tfidf_max_df,
        min_df=args.tfidf_min_df
    )
    
    K_best = SelectKBest(chi2, k=args.chi2_k)
    
    LR = LogisticRegression(
            C=args.lr_c,
            solver=args.lr_solver
        )
    return Pipeline([
        ('features', tfidf),
        ('chi2', K_best),
        ('classifier', LR)
    ])