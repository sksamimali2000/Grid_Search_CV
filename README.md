# 🔍 Hyperparameter Tuning with Grid Search (KNN & SVM)

A simple Python project that demonstrates **hyperparameter tuning** using `GridSearchCV` from **scikit-learn**  
for **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** classifiers on the classic **Iris dataset**.

---

## 🚀 Features

- Applies Grid Search for **KNN Classifier**  
- Applies Grid Search for **SVM Classifier**  
- Automatically finds the best hyperparameters for optimal accuracy  
- Simple and reproducible workflow using scikit-learn  

---

## 📊 Dataset

- Uses the classic **Iris Dataset** (`sklearn.datasets.load_iris`)  
- Dataset contains 150 samples of 3 Iris flower species with 4 features (sepal and petal dimensions)

---

## ⚡ Usage

```python
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Split into train & test
x_train, x_test, y_train, y_test = train_test_split(x, y)

# KNN Grid Search
clf = KNeighborsClassifier()
grid = {"n_neighbors": [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(clf, grid)
grid_search.fit(x_train, y_train)
print("Best KNN Estimator:", grid_search.best_estimator_)

# SVM Grid Search
clf = svm.SVC()
grid = {
    'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [1e-3, 5e-4, 1e-4, 5e-3]
}
grid_search = GridSearchCV(clf, grid)
grid_search.fit(x_train, y_train)
print("Best SVM Estimator:", grid_search.best_estimator_)
```







✅ Key Outcomes

Finds best n_neighbors for KNN based on grid search

Finds best C and gamma for SVM based on grid search

Improves model performance by tuning hyperparameters automatically

⚙️ Requirements

Python >= 3.7

scikit-learn

numpy

Install dependencies using:

pip install scikit-learn numpy

📄 License

MIT License

Made with ❤️ by Sk Samim Ali



