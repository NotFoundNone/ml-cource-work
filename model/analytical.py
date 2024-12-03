import numpy as np

import pandas as pd

data = pd.read_csv('../normalize/data/training/train_normalized_student_performance.csv')

# Разделим данные на признаки и целевую переменную
X = data.drop('Exam_Score', axis=1).values  # Признаки (X)
y = data['Exam_Score'].values  # Целевая переменная (y)

# Добавляем столбец единиц для смещения (intercept)
X = np.column_stack((np.ones(X.shape[0]), X))

X_T = X.T

theta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)

print("Коэффициенты модели:", theta)

np.savetxt('../normalize/data/theta/theta_analytic.csv', theta, delimiter=',', comments='')
