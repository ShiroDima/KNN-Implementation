from typing import Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean


class KNNClassifier:
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self._num_of_classes = None
        self.k = 3

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
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

    def indicator(self, j: int, y_i: int) -> bool:
        """
        This indicator function takes to inputs are returns true if they're the same and false if they not

        Parameters
        ----------
        j: An integer denoting the current class being compared
        y_i: An integer denoting the class of observation i

        Returns
        -------
        boolean
        """
        return j == y_i

    def _generate_k_neighbors(self, x: np.ndarray) -> np.array:
        """
        Generates an array of k closest neighbors to x in terms of the euclidean distance between them
        Parameters
        ----------
        x:np.array = the array of features to be classified

        Returns
        -------
        neighbors:np.array
        """
        try:
            n_cols = x.shape[1]
        except IndexError:
            raise IndexError(
                "If predicting on a single sample, use x.reshape(1, -1)")

        if self.X.shape[1] != n_cols:
            raise ValueError(
                "input x must have the same number of columns as the data matrix")

        if self.X.shape[1] < 2:
            distance = [euclidean(x_i, x[0]) for x_i in self.X]
            idx_array = np.argpartition(distance, self.k)
            neighbors = self.y[idx_array][:self.k].T

            return neighbors
        else:
            distance = []
            for j in range(x.shape[0]):
                row_distance = []
                for i in range(self.X.shape[0]):
                    row_distance.append(euclidean(self.X[i, :], x[j, :]))

                distance.append(row_distance)
            distance = np.array(distance)
            idx_array = np.argpartition(distance, self.k, axis=1)
            neighbors = np.take_along_axis(
                self.y, idx_array, axis=0)[:, :self.k]
            return neighbors

    def _generate_probs(self, y: np.ndarray) -> np.array:
        probs = np.zeros((y.shape[0], self._num_of_classes))
        for i in range(y.shape[0]):
            for j in range(self._num_of_classes):
                probs[i, j] = sum([self.indicator(j, y_j)
                                  for y_j in y[i, :]]) / self.k

        return probs

    def predict(self, x: np.ndarray, k: int = None) -> int:
        if self.X is None or self.y is None:
            raise TypeError(
                "Please fit the model before trying to use it for predictions")
        if k is not None:
            self.k = k
        neighbors = self._generate_k_neighbors(x)
        class_probabilities = self._generate_probs(neighbors)
        return np.argmax(class_probabilities, axis=1)


if __name__ == '__main__':
    # X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]).T
    # y = np.array([[0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]]).T
    class_data = pd.read_csv("../data/diabetes.csv")
    X, y = class_data.drop(columns=["Outcome"]), class_data[["Outcome"]]
    # print(isinstance(X, pd.DataFrame), isinstance(y, pd.DataFrame))
    # X_pred = X.iloc[3].values
    X_pred = X.iloc[5].values
    knn = KNNClassifier()
    knn.fit(X, y)
    print(knn.predict(X_pred))
