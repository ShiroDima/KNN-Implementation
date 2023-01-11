import pytest
import numpy as np
from regression.KNNRegressor import KNNRegressor


@pytest.fixture
def data():
    return np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]).T, np.array([[0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]]).T


@pytest.fixture
def KNN():
    return KNNRegressor()


@pytest.fixture
def FitKNN(data, KNN):
    KNN.fit(data[0], data[1])
    return KNN


def test_fit(data, FitKNN):
    assert np.equal(FitKNN.X, data[0]).all()
    assert np.equal(FitKNN.y, data[1]).all()


def test_generate_n_neighbors(FitKNN):
    assert max(FitKNN._generate_k_neighbors(np.array([[3]])).shape) == FitKNN.k


def test_will_not_predict_if_not_fit(KNN):
    with pytest.raises(TypeError):
        KNN.predict(np.array([[3]]), k=7)


def test_test_data_has_same_number_of_features(FitKNN):
    with pytest.raises(ValueError):
        FitKNN.predict(np.array([[3, 3]]))
