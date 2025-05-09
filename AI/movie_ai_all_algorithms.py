import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- ДАННЫЕ ---
movie_names = ["Inception", "Titanic", "Matrix", "Avengers", "Toy Story"]

users = {
    "User1": [5, 4, 5, 4, 1],
    "User2": [3, 5, 2, 3, 5],
    "User3": [4, 4, 4, 5, 2],
    "User4": [1, 2, 1, 2, 5],
    "TargetUser": [5, 0, 5, 0, 0]  # 0 — не оценено
}

# --- ОБЩИЕ ФУНКЦИИ ---
def prepare_data(users_dict):
    user_list = list(users_dict.keys())
    data = []
    for user in user_list:
        data.append(users_dict[user])
    return np.array(data), user_list

def get_training_data(users, target_film_index):
    X = []
    y = []
    for name, ratings in users.items():
        if name == "TargetUser":
            continue
        if ratings[target_film_index] != 0:
            features = ratings.copy()
            features.pop(target_film_index)
            X.append([f for i, f in enumerate(features) if i != target_film_index])
            y.append(1 if ratings[target_film_index] >= 4 else 0)  # бинарная классификация
    return np.array(X), np.array(y)

def get_target_features(target_ratings, exclude_index):
    features = target_ratings.copy()
    features.pop(exclude_index)
    return np.array([features])

# --- 1. KNN РЕКОМЕНДАЦИЯ ---
def knn_recommend(users_dict, k=2):
    data, user_list = prepare_data(users_dict)
    target_index = user_list.index("TargetUser")
    target_ratings = data[target_index]
    user_ratings = np.delete(data, target_index, axis=0)
    mask = target_ratings != 0
    filtered_users = user_ratings[:, mask]
    filtered_target = target_ratings[mask].reshape(1, -1)
    model = NearestNeighbors(n_neighbors=k, metric='cosine')
    model.fit(filtered_users)
    distances, indices = model.kneighbors(filtered_target)

    recommendations = {}
    for i, movie in enumerate(movie_names):
        if target_ratings[i] == 0:
            total = 0
            count = 0
            for idx in indices[0]:
                neighbor_rating = user_ratings[idx][i]
                if neighbor_rating != 0:
                    total += neighbor_rating
                    count += 1
            if count > 0:
                recommendations[movie] = round(total / count, 2)

    return dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))

# --- ОБУЧАЮЩАЯ ФУНКЦИЯ ДЛЯ КЛАССИФИКАТОРОВ ---
def train_and_predict(model, name, film_index):
    X, y = get_training_data(users, film_index)
    model.fit(X, y)
    target_features = get_target_features(users["TargetUser"], film_index)
    prediction = model.predict(target_features)[0]
    result = "понравится" if prediction else "не понравится"
    print(f"{name}: {result} фильм {movie_names[film_index]}")

# --- ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    print("\n=== 1. KNN Рекомендации ===")
    knn_result = knn_recommend(users)
    for movie, rating in knn_result.items():
        print(f"{movie}: вероятная оценка {rating}")

    film_index = movie_names.index("Avengers")  # проверяем на "Avengers"

    print("\n=== 2. Linear Regression ===")
    X = []
    y = []
    for name, ratings in users.items():
        if name == "TargetUser":
            continue
        if ratings[film_index] != 0:
            features = ratings.copy()
            features.pop(film_index)
            X.append(features)
            y.append(ratings[film_index])
    X = np.array(X)
    y = np.array(y)
    target_features = np.array([users["TargetUser"][:film_index] + users["TargetUser"][film_index+1:]])
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    pred = lr_model.predict(target_features)[0]
    print(f"Linear Regression: прогноз оценки фильма {movie_names[film_index]} — {round(pred, 2)}")

    print("\n=== 3. Logistic Regression ===")
    train_and_predict(LogisticRegression(), "Logistic Regression", film_index)

    print("\n=== 4. Decision Tree ===")
    dt_model = DecisionTreeClassifier(max_depth=3)
    train_and_predict(dt_model, "Decision Tree", film_index)

    # Визуализация дерева
    print("Открывается дерево решений...")
    plt.figure(figsize=(10, 5))
    plot_tree(dt_model, feature_names=[m for i, m in enumerate(movie_names) if i != film_index], class_names=["Не нравится", "Нравится"], filled=True)
    plt.show()

    print("\n=== 5. Random Forest ===")
    train_and_predict(RandomForestClassifier(n_estimators=100), "Random Forest", film_index)

    print("\n=== 6. Naive Bayes ===")
    train_and_predict(GaussianNB(), "Naive Bayes", film_index)

    print("\n=== 7. SVM ===")
    train_and_predict(SVC(kernel='linear'), "SVM", film_index)

    print("\n=== 8. Gradient Boosting ===")
    train_and_predict(GradientBoostingClassifier(), "Gradient Boosting", film_index)
