import pytest
import numpy as np
import pandas as pd
from classification.KNNClassifier import KNNClassifier


@pytest.fixture
def data():
    class_data = pd.read_csv("../data/diabetes.csv")
    X, y = class_data.drop(columns=["Outcome"]), class_data[["Outcome"]]
    return X, y


@pytest.fixture
def KNN():
    return KNNClassifier()


@pytest.fixture
def FitKNN(data, KNN):
    KNN.fit(data[0], data[1])
    return KNN


def test_fit(data, FitKNN):
    assert np.equal(FitKNN.X, data[0].values).all()
    assert np.equal(FitKNN.y, data[1].values).all()


def test_generate_k_neighbors(data, FitKNN):
    X_pred = data[0].iloc[:5].values
    assert FitKNN._generate_k_neighbors(X_pred).shape[1] == FitKNN.k


def test_indicator_function(KNN):
    assert KNN.indicator(1, 1)


def test_will_not_predict_if_not_fit(KNN):
    with pytest.raises(TypeError):
        KNN.predict(np.array([[3]]), k=7)


def test_test_data_has_same_number_of_features(FitKNN):
    with pytest.raises(ValueError):
        FitKNN.predict(np.array([[3, 3]]))

def test_data_has_correct_shape(data, FitKNN):
    X_pred = data[0].iloc[5].values
    with pytest.raises(IndexError):
        FitKNN.predict(X_pred)

