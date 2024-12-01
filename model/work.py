import numpy as np

# Загружаем полученные параметры theta
theta = np.loadtxt('optimized_theta.csv', delimiter=',')  # Загружаем оптимизированные параметры

# Данные для одного студента (предоставленные вами)
student_data = [0.5116279069767442,0.6,0.0,1.0,0.0,0.5,0.46,0.0,1.0,0.0,0.0,0.5,0.0,1.0,0.5,0.0,0.0,0.0,0.0]


# Преобразуем данные студента в массив
X_student = np.array(student_data).reshape(1, -1)

# Добавляем единичный столбец для учёта смещения (bias)
X_student = np.c_[np.ones(X_student.shape[0]), X_student]  # Добавляем единичный столбец

# Функция для предсказания
def predict(X, theta):
    return np.dot(X, theta)

# Предсказание для этого студента
predicted_exam_score = predict(X_student, theta)

# Выводим предсказанное значение
print(f'Предсказанный балл экзамена для студента: {predicted_exam_score[0]}')
