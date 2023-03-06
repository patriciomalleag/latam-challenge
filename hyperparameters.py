from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LGBMClassifier': LGBMClassifier()
}

params = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    },
    'LGBMClassifier': {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5, 10],
        'subsample': [0.5, 0.8],
        'colsample_bytree': [0.5, 0.8],
        'scale_pos_weight': [1, 2.5, 4]
    }
}