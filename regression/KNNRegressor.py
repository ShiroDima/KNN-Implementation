from typing import Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

class KNNRegressor:
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.k = 3


    def fit(self, X:Union[np.ndarray, pd.DataFrame], y:Union[np.ndarray, pd.Series]) -> None :
        """
        Stores the X and y values as numpy arrays to be used for predictions.
        Parameters
        ----------
        X: data matrix of features
        y: class labels as integers

        Returns
        -------
        None
        """
        if isinstance(X, pd.DataFrame) and (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            self.X = X.values
            self.y = y.values
            self._num_of_classes = len(np.unique(self.y))
        else:
            self.X = X
            self.y = y
            self._num_of_classes = len(np.unique(self.y))


    def _generate_k_neighbors(self, x:np.ndarray) -> np.array:
        """
        Generates an array of k closest neighbors to x in terms of the euclidean distance between them
        Parameters
        ----------
        x:np.array = the array of features to be classified

        Returns
        -------
        neighbors:np.array
        """
        if self.X.shape[1] != x.shape[1]:
            raise ValueError("input x must have the same number of columns as the data matrix")

        if self.X.shape[1] < 2:
            distance = [euclidean(x_i, x[0]) for x_i in self.X]
        else:
            distance = []
            for i in range(self.X.shape[0]):
                distance.append(euclidean(X[i, :], x))

        idx_array = np.argpartition(distance, self.k)
        neighbors = self.y[idx_array][:self.k].T

        return neighbors


    def _find_average(self, y:np.ndarray) -> np.array:
        return y.sum() / self.k


    def predict(self, x:np.ndarray, k:int=None) -> int:
        if self.X is None or self.y is None:
            raise TypeError("Please fit the model before trying to use it for predictions")
        if k is not None:
            self.k = k
        neighbors =  self._generate_k_neighbors(x)
        prediction = self._find_average(neighbors)
        return prediction


if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]).T
    y = np.array([[0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]]).T

    knn = KNNRegressor()
    knn.fit(X, y)
    print(knn.predict(np.array([[3]]), k=7))
