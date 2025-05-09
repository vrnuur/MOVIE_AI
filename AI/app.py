from flask import Flask, render_template, request, jsonify
from ai.recommender import knn_recommend, predict_preferences

app = Flask(__name__)

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Рекомендации на основе KNN (если используешь такую кнопку/маршрут)
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    users = {
        "User1": [5, 4, 5, 4, 1],
        "User2": [3, 5, 2, 3, 5],
        "User3": [4, 4, 4, 5, 2],
        "User4": [1, 2, 1, 2, 5],
        "TargetUser": list(data.values())
    }
    result = knn_recommend(users)
    return jsonify(result)

# Предсказание: понравится/не понравится
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    users = {
        "User1": [5, 4, 5, 4, 1],
        "User2": [3, 5, 2, 3, 5],
        "User3": [4, 4, 4, 5, 2],
        "User4": [1, 2, 1, 2, 5],
        "TargetUser": list(data.values())
    }

    # Индексы фильмов, которые пользователь не оценил (0)
    film_indices = [i for i, r in enumerate(users["TargetUser"]) if r == 0]

    # Предсказание для этих фильмов
    predictions = predict_preferences(users, film_indices)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)