import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загрузка данных
data = pd.read_csv('./test/predictions.csv')

# Разделение данных
actual_scores = data['Actual_Exam_Score']
predicted_grad_descent = data['Predicted_Exam_Score']
predicted_analytical = data['Predicted_Analytical_Exam_Score']
predicted_sklearn = data['Predicted_With_Lib_Exam_Score']

# Функция для расчёта метрик
def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

# Расчёт метрик для каждой модели
metrics_grad_descent = calculate_metrics(actual_scores, predicted_grad_descent)
metrics_analytical = calculate_metrics(actual_scores, predicted_analytical)
metrics_sklearn = calculate_metrics(actual_scores, predicted_sklearn)

# Вывод результатов
print("Метрики для модели градиентного спуска:", metrics_grad_descent)
print("Метрики для аналитической модели:", metrics_analytical)
print("Метрики для модели Scikit-learn:", metrics_sklearn)

# Анализ отклонений
def calculate_deviations(actual, predicted):
    deviations = np.abs(actual - predicted)
    avg_deviation = deviations.mean()
    max_deviation = deviations.max()
    return avg_deviation, max_deviation

# Расчёт отклонений
deviation_grad_descent = calculate_deviations(actual_scores, predicted_grad_descent)
deviation_analytical = calculate_deviations(actual_scores, predicted_analytical)
deviation_sklearn = calculate_deviations(actual_scores, predicted_sklearn)

# Вывод отклонений
print("\nСредние и максимальные отклонения для моделей:")
print(f"Градиентный спуск - Среднее отклонение: {deviation_grad_descent[0]:.2f}, Максимальное отклонение: {deviation_grad_descent[1]:.2f}")
print(f"Аналитическая модель - Среднее отклонение: {deviation_analytical[0]:.2f}, Максимальное отклонение: {deviation_analytical[1]:.2f}")
print(f"Scikit-learn - Среднее отклонение: {deviation_sklearn[0]:.2f}, Максимальное отклонение: {deviation_sklearn[1]:.2f}")

# График для визуализации предсказаний
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actual_scores, label="Фактические оценки", color='black', linewidth=2)
plt.plot(predicted_grad_descent, label="Градиентный спуск", linestyle='--', color='blue')
plt.plot(predicted_analytical, label="Аналитическая модель", linestyle='--', color='green')
plt.plot(predicted_sklearn, label="Scikit-learn", linestyle='--', color='red')
plt.xlabel("Итерации")
plt.ylabel("Оценки")
plt.title("Сравнение фактических и предсказанных значений")
plt.legend()
plt.show()
