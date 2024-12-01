import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv('data/StudentPerformanceFactors.csv')

# Разделим данные на числовые и категориальные, исключая столбец 'Exam_Score'
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Исключаем 'Exam_Score' из числовых столбцов
numerical_columns.remove('Exam_Score')

# Обработка числовых данных
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Заменяем пропуски на среднее значение
    ('scaler', StandardScaler())  # Стандартизируем
])

# Обработка категориальных данных
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заменяем пропуски на наиболее частое значение
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encoding
])

# Преобразование для всего датасета, исключая 'Exam_Score'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Применяем преобразования
data_transformed = preprocessor.fit_transform(data)

# Добавляем 'Exam_Score' в данные без изменений
exam_score = data['Exam_Score'].values.reshape(-1, 1)

# Объединяем преобразованные данные и столбец 'Exam_Score'
data_transformed_combined = np.hstack([data_transformed, exam_score])

# Если необходимо вернуть результат в DataFrame (с новыми столбцами для категориальных данных)
columns_transformed = numerical_columns + \
    preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_columns).tolist() + ['Exam_Score']

# Создаём DataFrame с новыми именами столбцов
data_transformed_df = pd.DataFrame(data_transformed_combined, columns=columns_transformed)

# Сохраняем результат
data_transformed_df.to_csv('processed_dataset.csv', index=False)

# Выводим первые строки обработанного датасета
print(data_transformed_df.head())
