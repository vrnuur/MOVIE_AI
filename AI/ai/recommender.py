from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import numpy as np

movie_names = ["Inception", "Titanic", "Matrix", "Avengers", "Toy Story"]

def knn_recommend(users, target='TargetUser', n_neighbors=2):
    X = np.array(list(users.values()))
    model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    model.fit(X)
    distances, indices = model.kneighbors([users[target]])
    
    recommended = {}
    for idx in indices[0][1:]:
        neighbor = list(users.keys())[idx]
        for i, rating in enumerate(users[neighbor]):
            if users[target][i] == 0:
                recommended[movie_names[i]] = rating
    return recommended

# helper functions
def get_training_data(users_dict, film_index):
    X, y = [], []
    for user, ratings in users_dict.items():
        if user == "TargetUser":
            continue
        rating = ratings[film_index]
        if rating != 0:
            features = [r for i, r in enumerate(ratings) if i != film_index]
            X.append(features)
            y.append(rating if isinstance(rating, float) or isinstance(rating, int) else 0)
    return X, y

def get_target_features(target_ratings, film_index):
    return [[r for i, r in enumerate(target_ratings) if i != film_index]]

def predict_preferences(users_dict, film_indices):
    results = {}
    for film_index in film_indices:
        movie = movie_names[film_index]
        results[movie] = {}

        X, y = get_training_data(users_dict, film_index)
        if len(X) == 0:
            results[movie]["error"] = "Недостаточно данных"
            continue

        target = get_target_features(users_dict["TargetUser"], film_index)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        pred_score = lr.predict(target)[0]
        results[movie]["Linear Regression"] = f"{round(pred_score, 2)}"

        # Бинарные классификаторы (1 если >= 3, иначе 0)
        y_binary = [1 if val >= 3 else 0 for val in y]
        classifiers = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(max_depth=3),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(kernel='linear', probability=True),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        for name, clf in classifiers.items():
            clf.fit(X, y_binary)
            prediction = clf.predict(target)[0]
            results[movie][name] = "Понравится" if prediction == 1 else "Не понравится"
    return results