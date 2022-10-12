import numpy as np


def calculate_rvce(y_true: np.array, y_pred: np.array):
    y_true_sum = y_true.sum()
    return np.abs(y_pred.sum() - y_true_sum) / y_true_sum


def calculate_rvce_of_svm_classifier(classifier, x: np.array, y_true: np.array):
    return calculate_rvce(y_true, classifier.predict(x))
