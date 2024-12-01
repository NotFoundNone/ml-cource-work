import numpy as np

import pandas as pd

data = pd.read_csv('../normalize/data/normalized_student_performance.csv')

# Разделим данные на признаки и целевую переменную
X = data.drop('Exam_Score', axis=1).values  # Признаки (X)
y = data['Exam_Score'].values  # Целевая переменная (y)

# Добавляем столбец единиц для смещения (перехвата)
X = np.column_stack((np.ones(X.shape[0]), X))

X_T = X.T

theta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)

print("Коэффициенты модели:", theta)

np.savetxt('../normalize/data/theta/theta_analytic.csv', theta, delimiter=',', comments='')


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