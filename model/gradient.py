import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Пример загрузки датасета (замените на свой путь)
data = pd.read_csv('../normalize/data/training/train_normalized_student_performance.csv')

# Разделим данные на признаки и целевую переменную
X = data.drop('Exam_Score', axis=1).values  # Признаки (X)
y = data['Exam_Score'].values  # Целевая переменная (y)

# Добавляем единичный столбец для учёта смещения (bias) в уравнении
X = np.c_[np.ones(X.shape[0]), X]

plt.boxplot(X)
plt.show()

# Инициализация параметров модели (веса)
theta = np.zeros(X.shape[1])


# Функция для предсказания
def predict(X, theta):
    return np.dot(X, theta)

# Функция для вычисления средней квадратичной ошибки (MSE)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # Вычисление градиента
        predictions = predict(X, theta)
        gradients = (1 / m) * np.dot(X.T, (predictions - y))

        # Проверяем на большие градиенты
        if np.any(np.abs(gradients) > 1e10):
            print(f"Градиенты слишком большие на итерации {i}, остановка обучения.")
            break

        # Обновление параметров
        theta = theta - alpha * gradients

        # Вычисление стоимости для отслеживания
        cost_history.append(compute_cost(X, y, theta))
        if i % 100 == 0: # Логируем каждую сотую итерацию
            print(f"Итерация {i}, Стоимость: {cost_history[-1]}")

    return theta, cost_history

# Разные скорости обучения для тестирования
alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.07,  0.1, 0.2, 0.5, 1.0]
iterations = 1000  # Количество итераций

# Словарь для хранения стоимости для разных alpha
costs = {}

start_train = time.time()
# Протестируем каждое значение alpha
for alpha in alphas:
    theta = np.zeros(X.shape[1])  # Инициализация параметров
    optimized_theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

    if len(cost_history) == iterations:
        costs[alpha] = cost_history

        # График изменения стоимости для каждого alpha
        plt.plot(range(1, iterations + 1), cost_history, label=f'alpha={alpha}')
    else:
        print(f"Прерывание на alpha={alpha} из-за больших градиентов.")

end_train = time.time()

train_time = end_train - start_train

print("train time: ", train_time)

np.savetxt('../normalize/data/theta/optimized_theta.csv', optimized_theta, delimiter=',', comments='')

# Настройки графика
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent with Different Alpha values')
plt.legend()
plt.show()

# Определяем лучший alpha, с минимальной конечной стоимостью
final_costs = {alpha: cost_history[-1] for alpha, cost_history in costs.items()}
best_alpha = min(final_costs, key=final_costs.get)

print(f"Лучший alpha: {best_alpha} с финальной стоимостью {final_costs[best_alpha]}")
