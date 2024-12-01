import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Маппинг для категориальных данных
category_mappings = {
    'Low': 0, 'Medium': 1, 'High': 2,
    'No': 0, 'Yes': 1,
    'Public': 0, 'Private': 1,
    'Positive': 2, 'Neutral': 1, 'Negative': 0,
    'Near': 0, 'Moderate': 1, 'Far': 2,
    'Male': 0, 'Female': 1,
    'High School': 0, 'College': 1, 'Postgraduate': 2
}

# Функция для преобразования категориальных данных в числовые
def transform_categories(data, mappings):
    transformed_data = []
    for row in data:
        transformed_row = []
        for val in row:
            if val == '' or val is None:  # Обработка пустых строк или значений
                transformed_row.append(0)  # Можно заменить на 0 или другое значение по умолчанию
            elif val in mappings:
                transformed_row.append(mappings[val])  # Преобразуем категориальные данные
            else:
                try:
                    transformed_row.append(float(val))  # Преобразуем в число
                except ValueError:
                    transformed_row.append(0)  # Для любых ошибок преобразования можно задать значение по умолчанию
        transformed_data.append(transformed_row)
    return np.array(transformed_data)

# Загрузка данных из CSV
data = pd.read_csv('data/StudentPerformanceFactors.csv')

# Преобразуем данные в numpy массив
data_transformed = transform_categories(data.values, category_mappings)

normalized_data = pd.DataFrame(data_transformed, columns=data.columns)

normalized_data.to_csv('proverka.csv', index=False)

# # Разделим данные на признаки и целевую переменную
# X = data_transformed[:, :-1]  # Признаки (X) - все столбцы, кроме последнего
# y = data_transformed[:, -1]   # Целевая переменная (y) - последний столбец
#
# # Нормализация данных
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Преобразуем нормализованные данные обратно в DataFrame с соответствующими названиями
# normalized_data = pd.DataFrame(X_scaled, columns=data.columns[:-1])  # Без целевой переменной
#
# # Добавляем целевую переменную в DataFrame
# normalized_data['Exam_Score'] = y
#
# # Запись нормализованных данных в CSV файл
# normalized_data.to_csv('proverka.csv', index=False)


# # Разделим данные на числовые и категориальные, исключая столбец 'Exam_Score'
# numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
# categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
#
# # Исключаем 'Exam_Score' из числовых столбцов
# numerical_columns.remove('Exam_Score')
#
# # Обработка числовых данных
# numerical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),  # Заменяем пропуски на среднее значение
#     ('scaler', StandardScaler())  # Стандартизируем
# ])
#
# # Обработка категориальных данных
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),  # Заменяем пропуски на наиболее частое значение
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encoding
# ])
#
# # Преобразование для всего датасета, исключая 'Exam_Score'
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_columns),
#         ('cat', categorical_transformer, categorical_columns)
#     ])
#
# # Применяем преобразования
# data_transformed = preprocessor.fit_transform(data)
#
# # Добавляем 'Exam_Score' в данные без изменений
# exam_score = data['Exam_Score'].values.reshape(-1, 1)
#
# # Объединяем преобразованные данные и столбец 'Exam_Score'
# data_transformed_combined = np.hstack([data_transformed, exam_score])
#
# # Если необходимо вернуть результат в DataFrame (с новыми столбцами для категориальных данных)
# columns_transformed = numerical_columns + \
#     preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_columns).tolist() + ['Exam_Score']
#
# # Создаём DataFrame с новыми именами столбцов
# data_transformed_df = pd.DataFrame(data_transformed_combined, columns=columns_transformed)
#
# # Сохраняем результат
# data_transformed_df.to_csv('processed_dataset.csv', index=False)
#
# # Выводим первые строки обработанного датасета
# print(data_transformed_df.head())
