# MOVIE_AI
html/css/js/py/ai

# Movie Recommender System with Machine Learning

Этот проект — интеллектуальная система рекомендаций фильмов на основе оценок пользователя, реализованная с использованием 12 популярных алгоритмов машинного обучения.

##  О проекте

Вы вводите свои оценки для пяти популярных фильмов, а система:

- Рекомендует фильмы, похожие на ваши вкусы с помощью **KNN**
- Предсказывает, понравятся ли вам фильмы, используя:
  -  Линейную и логистическую регрессию
  -  Деревья решений и случайный лес
  -  Байесовский классификатор
  -  SVM, градиентный бустинг и другие

## Используемые алгоритмы

1. Linear Regression  
2. Logistic Regression  
3. Decision Tree  
4. Random Forest  
5. Naive Bayes Classifier  
6. K-Nearest Neighbor (KNN)  
7. Support Vector Machine (SVM)  
8. Gradient Boosting  
9. K-Means Clustering  
10. Apriori Algorithm  
11. Principal Component Analysis (PCA)  
12. Computer Vision (в будущем)

##  Интерфейс

Простой веб-интерфейс с формой ввода и двумя кнопками:
- **KNN Recommendations** — получить фильмы, которые вам могут понравиться
- **Will I Like It?** — оценка по алгоритмам

##  Установка и запуск

1. Клонируйте репозиторий:

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
````

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Запустите сервер (например, Flask):

```bash
python app.py
```

4. Перейдите в браузер:

```
http://localhost:5000
```

##  Структура проекта

```
movie-recommender/
│
├── app.py                # Основной Python-сервер
├── models.py             # Алгоритмы машинного обучения
├── templates/
│   └── index.html        # Веб-интерфейс
├── static/
│   └── style.css         # Стили CSS
├── README.md
└── requirements.txt
```

## 📌 Примечание

* Проект создан в учебных целях.
* Алгоритмы работают на заранее заданных данных.
* Возможна интеграция с базой данных и расширение функциональности.


