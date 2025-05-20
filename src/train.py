import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import save_model

X_train, X_test, y_train, y_test, preprocessor = pd.read_pickle('data/processed/train_test.pkl')

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'MultinomialNB': MultinomialNB()
}

results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probs)
        }
    except Exception as e:
        print(f"{name} failed: {e}")

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model = models[best_name]
print(f"Best model: {best_name} with ROC-AUC={results[best_name]['roc_auc']}")

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

pipeline.fit(X_train, y_train)
save_model(pipeline)