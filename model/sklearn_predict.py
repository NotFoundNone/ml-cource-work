import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Загрузка данных
data = pd.read_csv('../normalize/data/StudentPerformanceFactors.csv')

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('Exam_Score', axis=1)
y = data['Exam_Score']

# Обработка пропусков
# Для числовых признаков используем медиану, для категориальных - моду
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Применяем обработку пропусков
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = SimpleImputer(strategy='most_frequent')

# Кодирование категориальных признаков (One-Hot Encoding)
categorical_pipeline = Pipeline(steps=[
    ('imputer', categorical_transformer),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Нормализация и стандартизация для числовых признаков
numerical_pipeline = Pipeline(steps=[
    ('imputer', numerical_transformer),
    ('scaler', StandardScaler())  # или MinMaxScaler() для нормализации
])

# Объединение всех преобразований в одном Pipeline с ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

# Создание модели с обработкой данных
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2: {r2}")

new_data = pd.DataFrame({
    'Hours_Studied': [23],
    'Attendance': [84],
    'Parental_Involvement': ['Low'],
    'Access_to_Resources': ['High'],
    'Extracurricular_Activities': ['No'],
    'Sleep_Hours': [7],
    'Previous_Scores': [73],
    'Motivation_Level': ['Low'],
    'Internet_Access': ['Yes'],
    'Tutoring_Sessions': [0],
    'Family_Income': ['Low'],
    'Teacher_Quality': ['Medium'],
    'School_Type': ['Public'],
    'Peer_Influence': ['Positive'],
    'Physical_Activity': [3],
    'Learning_Disabilities': ['No'],
    'Parental_Education_Level': ['High School'],
    'Distance_from_Home': ['Near'],
    'Gender': ['Male']
})

# Прогнозирование с использованием ранее обученной модели
y_pred_test = model.predict(new_data)

# Выводим результат
print(f"Прогнозируемая оценка экзамена: {y_pred_test[0]}")