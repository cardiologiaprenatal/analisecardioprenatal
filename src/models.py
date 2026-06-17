"""Definicao e factory de modelos de Machine Learning."""

from __future__ import annotations
from typing import Any
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.config import RANDOM_STATE

N_EST = 80

def get_model_registry(class_weight=None):
    cw = class_weight
    svm = CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE), cv=3)
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=cw, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=N_EST, max_depth=10, random_state=RANDOM_STATE, class_weight=cw, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=N_EST, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="logloss", scale_pos_weight=(cw[1]/cw[0]) if cw else 1.0, verbosity=0, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, class_weight=cw, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=N_EST, depth=5, learning_rate=0.1, random_state=RANDOM_STATE, auto_class_weights="Balanced" if cw else None, verbose=0, allow_writing_files=False, thread_count=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=N_EST, max_depth=10, random_state=RANDOM_STATE, class_weight=cw, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=60, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE, class_weight=cw),
        "SVM": svm,
        "KNN": KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1),
    }