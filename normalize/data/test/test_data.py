import numpy as np
import pandas as pd

# Загружаем полученные параметры theta
theta = np.loadtxt('../theta/optimized_theta.csv', delimiter=',')  # Загружаем оптимизированные параметры
analytical_theta = np.loadtxt('../theta/theta_analytic.csv', delimiter=',')

# Загружаем тестовые данные из CSV файла
test_data = pd.read_csv('test_normalized_student_performance.csv')  # Замените путь на путь к вашему тестовому файлу

# Разделяем на признаки (X) и целевую переменную (y)
X_test = test_data.drop(columns=['Exam_Score'])  # Убираем столбец с целевой переменной
y_test = test_data['Exam_Score']  # Столбец с реальными значениями экзаменов

# Преобразуем данные в формат numpy массивов
X_test = X_test.to_numpy()

# Добавляем единичный столбец для учёта смещения (bias)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Добавляем единичный столбец

# Функция для предсказания
def predict(X, theta):
    return np.dot(X, theta)

# Предсказания для всей тестовой выборки
predicted_exam_scores = predict(X_test, theta)
analytical_exam_scores = predict(X_test,analytical_theta)

# Выводим реальные и предсказанные значения
for i in range(len(y_test)):
    print(f'Реальный балл: {y_test.iloc[i]}, Предсказанный балл: {predicted_exam_scores[i]}')

# Создаем DataFrame для записи в файл
results = pd.DataFrame({
    'Actual_Exam_Score': y_test,  # Реальные баллы
    'Predicted_Exam_Score': predicted_exam_scores,  # Предсказанные баллы
    'Predicted_Analytical_Exam_Score': analytical_exam_scores # Предсказанные баллы при помощи аналитического способа
})

# Сохраняем результат в CSV файл
results.to_csv('predictions.csv', index=False)
