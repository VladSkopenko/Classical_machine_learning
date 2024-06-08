from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
X, y = load_breast_cancer(return_X_y=True)

clf = RidgeClassifier(alpha=30.0).fit(X, y)
clf.score(X, y)
