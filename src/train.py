import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
from sklearn.naive_bayes import GaussianNB

X_train_p, X_test_p, y_train, y_test, preprocessor, X_train_raw, X_test_raw = pd.read_pickle('data/processed/train_test.pkl')



models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    "GaussianNB": GaussianNB() 
}


best_model = None
best_score = 0

for name, model in models.items():
    try:
        model.fit(X_train_p, y_train)
        y_pred_proba = model.predict_proba(X_test_p)[:, 1]
        score = roc_auc_score(y_test, y_pred_proba)
        print(f"{name} ROC-AUC: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
    except Exception as e:
        print(f"{name} failed: {e}")

print(f"Best model: {best_model.__class__.__name__} with ROC-AUC={best_score:.4f}")

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

pipeline.fit(X_train_raw, y_train)

joblib.dump(pipeline, 'models/final_model.pkl')
