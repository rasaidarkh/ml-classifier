import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Загрузка данных
df = pd.read_csv('data/raw/heart.csv')  # укажи своё имя файла

# Разделение на X и y
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Указание признаков
numerical = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Препроцессор
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Тренировочная и тестовая выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем данные
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# Сохраняем всё
pd.to_pickle(
    (X_train_p, X_test_p, y_train, y_test, preprocessor, X_train, X_test),
    'data/processed/train_test.pkl'
)
