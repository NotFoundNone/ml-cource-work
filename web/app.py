import pandas as pd
from flask import Flask, render_template, request
import joblib
import numpy as np

from normalize.normalize import normalization_func
from normalize.standartization import transform_categories, category_mappings

def standardize_value(value, mean, std):
    return (value - mean) / std

def load_mean_std_values():
    mean_std_df = pd.read_csv('../normalize/data/util/mean_std_values.csv')
    mean_std_df.columns = mean_std_df.columns.str.strip().str.lower().str.replace(' ', '_')
    mean_values = mean_std_df.set_index('parameter')['mean'].to_dict()
    std_values = mean_std_df.set_index('parameter')['std'].to_dict()
    return mean_values, std_values

# Создание Flask-приложения
app = Flask(__name__)

# Маршрут для главной страницы с формой
@app.route('/')
def home():
    return render_template('index.html')

# Маршрут для получения предсказания
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем параметры из формы
        hours_studied = float(request.form['Hours_Studied'])
        attendance = float(request.form['Attendance'])
        parental_involvement = request.form['Parental_Involvement']
        access_to_resources = request.form['Access_to_Resources']
        extracurricular_activities = request.form['Extracurricular_Activities']
        sleep_hours = float(request.form['Sleep_Hours'])
        previous_scores = float(request.form['Previous_Scores'])
        motivation_level = request.form['Motivation_Level']
        internet_access = request.form['Internet_Access']
        tutoring_sessions = float(request.form['Tutoring_Sessions'])
        family_income = request.form['Family_Income']
        teacher_quality = request.form['Teacher_Quality']
        school_type = request.form['School_Type']
        peer_influence = request.form['Peer_Influence']
        physical_activity = float(request.form['Physical_Activity'])
        learning_disabilities = request.form['Learning_Disabilities']
        parental_education_level = request.form['Parental_Education_Level']
        distance_from_home = request.form['Distance_from_Home']
        gender = 1 if request.form['Gender'] == 'Male' else 0  # 1 for Male, 0 for Female

        # Формируем список значений для предсказания
        input_features = np.array([[
            hours_studied,
            attendance,
            parental_involvement,
            access_to_resources,
            extracurricular_activities,
            sleep_hours,
            previous_scores,
            motivation_level,
            internet_access,
            tutoring_sessions,
            family_income,
            teacher_quality,
            school_type,
            peer_influence,
            physical_activity,
            learning_disabilities,
            parental_education_level,
            distance_from_home,
            gender
        ]])

        transformed_data = transform_categories(input_features, category_mappings)

        parameters = [
            'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
            'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
            'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality', 'School_Type',
            'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities', 'Parental_Education_Level',
            'Distance_from_Home', 'Gender'
        ]

        # Преобразование np.array в словарь
        input_features_dict = {param: transformed_data[0][i] for i, param in enumerate(parameters)}

        input_features_dict_python_types = {key: value.item() if isinstance(value, np.generic) else value
                                            for key, value in input_features_dict.items()}

        mean_values , std_values = load_mean_std_values()

        # Стандартизация значений
        standardized_features = {}
        for parameter, value in input_features_dict_python_types.items():
            mean = mean_values.get(parameter, 0)  # Получаем среднее значение, если параметр найден
            std = std_values.get(parameter, 1)    # Получаем std значение, если параметр найден
            standardized_value = standardize_value(value, mean, std)  # Стандартизируем значение
            standardized_features[parameter] = standardized_value

        # Выводим стандартизированные значения
        print("!!!", standardized_features)

        standardized_values = np.array(list(standardized_features.values()))

        standardized_values_2d = standardized_values.reshape(1, -1)

        X_student = np.c_[np.ones(standardized_values_2d.shape[0]), standardized_values_2d]

        theta = np.loadtxt('../normalize/data/theta/optimized_theta.csv', delimiter=',')

        prediction = np.dot(X_student, theta)

        prediction = max(0, min(prediction[0], 100))

        return render_template('index.html', prediction_text=f'Предполагаемая оценка: {prediction}')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
