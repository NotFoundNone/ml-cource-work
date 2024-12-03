import random

import pandas as pd

# Загружаем CSV файл в DataFrame с помощью pandas
encoded_df = pd.read_csv('StudentPerformanceFactors.csv')

# Функция для перемешивания строк DataFrame с помощью алгоритма Фишера-Йетса
def shuffle_dataframe_fisher_yates(df):
    # Преобразуем индексы строк DataFrame в список
    indices = list(df.index)

    # Проходим по каждому индексу от последнего до первого
    for i in range(len(indices) - 1, 0, -1):
        # Генерируем случайный индекс от 0 до текущего индекса i
        j = random.randint(0, i)

        # Меняем местами элементы с индексами i и j
        indices[i], indices[j] = indices[j], indices[i]

    # Возвращаем DataFrame с перемешанными строками на основе перемешанных индексов
    return df.loc[indices].reset_index(drop=True)

# Вызываем функцию для перемешивания строк датасета
shuffled_encoded_df = shuffle_dataframe_fisher_yates(encoded_df)

# Сохранение перемешанного датасета в CSV файл
shuffled_encoded_df.to_csv('StudentPerformanceFactorsShuffle.csv', index=False)
