import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Загрузка обучающих и тестовых данных из файлов
X_train = pd.read_csv('../normalize/data/training/train_normalized_student_performance.csv')
y_train = X_train['Exam_Score']
# Удаление столбца "Exam_Score" из X_train
X_train = X_train.drop('Exam_Score', axis=1)

X_test = pd.read_csv('../normalize/data/test/test_normalized_student_performance.csv')
y_test = X_test['Exam_Score']
# Удаление столбца "Exam_Score" из X_test
X_test = X_test.drop('Exam_Score', axis=1)


# Обработка пропусков
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

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

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Средняя абсолютная ошибка: {mae}")
print(f"Среднеквадратичная ошибка: {mse}")
print(f"Коэф детерминации (R^2): {r2}")


data = pd.read_csv('../normalize/data/test/predictions.csv')

data['Predicted_With_Lib_Exam_Score'] = y_pred

# Сохранение обновленного файла
data.to_csv('../normalize/data/test/predictions.csv', index=False)

