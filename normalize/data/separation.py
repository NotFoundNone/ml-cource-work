import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем данные в pandas DataFrame (предположим, что ваши данные сохранены в CSV)
data = pd.read_csv('StudentPerformanceFactorsShuffle.csv')

# Определяем признаки (features) и целевую переменную (target)
X = data.drop(columns=['Exam_Score'])  # Все колонки, кроме целевой переменной
y = data['Exam_Score']  # Целевая переменная

# Разделяем на обучающую и тестовую выборки (80% для обучения, 20% для теста)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Объединяем обучающую выборку (X_train и y_train) в один DataFrame
train_data = X_train.copy()
train_data['Exam_Score'] = y_train

# Объединяем тестовую выборку (X_test и y_test) в один DataFrame
test_data = X_test.copy()
test_data['Exam_Score'] = y_test

# Сохраняем в CSV файлы
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Выводим размеры выборок
print(f'Размер обучающей выборки: {train_data.shape[0]}')
print(f'Размер тестовой выборки: {test_data.shape[0]}')
