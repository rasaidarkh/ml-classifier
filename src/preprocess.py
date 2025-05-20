import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path: str):
    return pd.read_csv(path)


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


if __name__ == '__main__':
    df = load_data('data/raw/data.csv')
    df = df.drop_duplicates().dropna()
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    numeric_feats = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_feats = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    pd.to_pickle((X_train_p, X_test_p, y_train, y_test, preprocessor), 'data/processed/train_test.pkl')