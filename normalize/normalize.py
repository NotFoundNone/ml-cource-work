import pandas as pd

# Функция для стандартизации
def normalization_func(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    # Словарь для хранения средних значений и стандартных отклонений
    mean_std_values = {}

    for column in df.columns:
        if column not in exclude_columns:
            mean_val = df[column].mean()  # Среднее значение
            std_val = df[column].std()    # Стандартное отклонение

            # Сохраняем среднее и стандартное отклонение
            mean_std_values[column] = {'mean': mean_val, 'std': std_val}

            # Стандартизация столбца
            df[column] = (df[column] - mean_val) / std_val

    return df, mean_std_values

# #Для обучающей выборки
#
# # Чтение данных
# data = pd.read_csv('./data/training/train_standartized_student_performance.csv')
#
# # Столбец, который не должен быть стандартизован
# exclude_columns = ['Exam_Score']
#
# # Стандартизация данных
# standardized_data, mean_std_values = normalization_func(data.copy(), exclude_columns=exclude_columns)
#
# # Сохранение стандартизированных данных в новый файл
# standardized_data.to_csv("./data/training/train_normalized_student_performance.csv", index=False)
#
# # Сохранение средних значений и стандартных отклонений для каждого признака
# mean_std_df = pd.DataFrame(mean_std_values).T
#
# # Добавляем индекс как отдельный столбец, чтобы он записался в файл
# mean_std_df.reset_index(inplace=True)
# mean_std_df.rename(columns={'index': 'parameter'}, inplace=True)
#
# # Сохранение значений среднего и стандартного отклонения
# mean_std_df.to_csv('./data/util/train_mean_std_values.csv', index=False)
#
# #Для тестовой выборки
#
# # Чтение данных
# data = pd.read_csv('./data/test/test_standartized_student_performance.csv')
#
# # Столбец, который не должен быть стандартизован
# exclude_columns = ['Exam_Score']
#
# # Стандартизация данных
# standardized_data, mean_std_values = normalization_func(data.copy(), exclude_columns=exclude_columns)
#
# # Сохранение стандартизированных данных в новый файл
# standardized_data.to_csv("./data/test/test_normalized_student_performance.csv", index=False)
#
# # Сохранение средних значений и стандартных отклонений для каждого признака
# mean_std_df = pd.DataFrame(mean_std_values).T
#
# # Добавляем индекс как отдельный столбец, чтобы он записался в файл
# mean_std_df.reset_index(inplace=True)
# mean_std_df.rename(columns={'index': 'parameter'}, inplace=True)
#
# # Сохранение значений среднего и стандартного отклонения
# mean_std_df.to_csv('./data/util/test_mean_std_values.csv', index=False)


# import pandas as pd
# import os
#
# # Функция для Мин-Макс нормализации
# def min_max_normalization(df, exclude_columns=None):
#     if exclude_columns is None:
#         exclude_columns = []
#
#     # Словарь для хранения минимальных и максимальных значений
#     min_max_values = {}
#
#     # Применение нормализации ко всем столбцам, кроме исключенных
#     for column in df.columns:
#         if column not in exclude_columns:
#             min_val = df[column].min()
#             max_val = df[column].max()
#
#             # Сохраняем минимальное и максимальное значения
#             min_max_values[column] = {'min': min_val, 'max': max_val}
#
#             # Нормализация столбца
#             df[column] = (df[column] - min_val) / (max_val - min_val)
#
#     return df, min_max_values
#
# # Чтение данных
# data = pd.read_csv('proverka.csv')
#
# # Столбец, который не должен быть нормализован
# exclude_columns = ['Exam_Score']
#
# # Нормализация данных
# normalized_data, min_max_values = min_max_normalization(data.copy(), exclude_columns=exclude_columns)
#
# # Сохранение нормализованных данных в новый файл
# normalized_data.to_csv("norm.csv", index=False)
#
# # Сохранение минимальных и максимальных значений для каждого признака
# min_max_df = pd.DataFrame(min_max_values).T
#
# # Добавляем индекс как отдельный столбец, чтобы он записался в файл
# min_max_df.reset_index(inplace=True)
# min_max_df.rename(columns={'index': 'parameter'}, inplace=True)