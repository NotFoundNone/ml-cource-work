# import numpy as np
# import os
#
# import pandas as pd
#
# # data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'normalize', 'data'))
# data = pd.read_csv('../normalize/data/processed_dataset.csv')
#
#
#
# # Разделим данные на признаки и целевую переменную
# X = data.drop('Exam_Score', axis=1).values  # Признаки (X)
# y = data['Exam_Score'].values  # Целевая переменная (y)
#
# # столбец единиц
# X = np.c_[np.ones(X.shape[0]), X]
#
# # theta = (X^T * X)^(-1) * X^T * y
#
# X_T = X.T
# theta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
# np.savetxt('theta_analytic.csv', theta, delimiter=',', header='theta', comments='')
# print(theta)
#
import numpy as np
import csv
#
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

# Функция для загрузки данных из CSV
def load_data_from_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Пропускаем заголовок
        for line in csv_reader:
            data.append(line)
    return data

# Загрузка данных из CSV файла
file_path = '../normalize/norm.csv'  # Путь к вашему CSV файлу
data = load_data_from_csv(file_path)

# Преобразуем данные в numpy массив
data_transformed = transform_categories(data, category_mappings)

output_file_path = '../normalize/data/transformed_student_performance.csv'
np.savetxt(output_file_path, data_transformed, delimiter=',', fmt='%f', header="Hours_Studied,Attendance,Parental_Involvement,Access_to_Resources,Extracurricular_Activities,Sleep_Hours,Previous_Scores,Motivation_Level,Internet_Access,Tutoring_Sessions,Family_Income,Teacher_Quality,School_Type,Peer_Influence,Physical_Activity,Learning_Disabilities,Parental_Education_Level,Distance_from_Home,Gender,Exam_Score", comments='')


# Разделяем на признаки и целевую переменную
X = data_transformed[:, :-1]  # Все столбцы, кроме последнего (целевой переменной)
y = data_transformed[:, -1]   # Последний столбец - это целевая переменная (Exam_Score)

# Добавляем столбец единиц для смещения (перехвата)
X = np.column_stack((np.ones(X.shape[0]), X))

# Шаг 1: Транспонируем X
X_T = X.T

# Шаг 2: Вычисляем (X^T X)
X_T_X = X_T.dot(X)

# Шаг 3: Вычисляем обратную матрицу (X^T X)^-1
X_T_X_inv = np.linalg.inv(X_T_X)

# Шаг 4: Вычисляем X^T y
X_T_y = X_T.dot(y)

# Шаг 5: Находим коэффициенты theta
theta = X_T_X_inv.dot(X_T_y)

print("Коэффициенты модели:", theta)

np.savetxt('theta_analytic.csv', theta, delimiter=',', header='theta', comments='')
