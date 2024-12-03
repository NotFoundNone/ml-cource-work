import numpy as np
import pandas as pd

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

# #Для обучающей выборки
# # Загрузка данных из CSV
# data = pd.read_csv('data/training/train_data.csv')
#
# # Преобразуем данные в numpy массив
# data_transformed = transform_categories(data.values, category_mappings)
#
# normalized_data = pd.DataFrame(data_transformed, columns=data.columns)
#
# normalized_data.fillna(0, inplace=True)   # Заменяет NaN на 0
# print(normalized_data.isna().sum())
#
# normalized_data.to_csv('./data/training/train_standartized_student_performance.csv', index=False)
#
# #Для тестовой выборки
#
# # Загрузка данных из CSV
# data = pd.read_csv('data/test/test_data.csv')
#
# # Преобразуем данные в numpy массив
# data_transformed = transform_categories(data.values, category_mappings)
#
# normalized_data = pd.DataFrame(data_transformed, columns=data.columns)
#
# normalized_data.fillna(0, inplace=True)   # Заменяет NaN на 0
# print(normalized_data.isna().sum())
#
# normalized_data.to_csv('./data/test/test_standartized_student_performance.csv', index=False)
