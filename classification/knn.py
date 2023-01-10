from typing import Union, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

class KNN:
    def __init__(self) -> None:
        self.X = None,
        self.y = None,
        self._num_of_classes = None
        self.k = 3

    def fit(self, X:Union[np.ndarray, pd.DataFrame], y:Union[np.ndarray, pd.Series]) -> None:
        """Stores the X and y values as numpy arrays to be used for predictions."""
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            self.X = X.values
            self.y = y.values
            self._num_of_classes = len(np.unique(self.y))
        else:
            self.X = X
            self.y = y
            self._num_of_classes = len(np.unique(self.y))

    def indicator(self, j:int, y_i:int) -> bool:
        return j == y_i

    def _generate_n_neighbors(self, x:np.ndarray) -> np.array:
        # print(x)
        if self.X.shape[1] != x.shape[1]:
            raise ValueError("input x must have the same number of columns as the data matrix")

        if X.shape[1] < 2:
            distance = [euclidean(x_i, x[0]) for x_i in self.X]
        else:
            distance = []
            for i in range(self.X.shape[0]):
                distance.append(euclidean(X[i, :], x))

        idx_array = np.argpartition(distance, self.k)
        return self.y[idx_array][:self.k].T

    def _generate_probs(self, y:np.ndarray) -> np.array:
        probs = np.zeros((self._num_of_classes, 1))
        for j in range(self._num_of_classes):
            probs[j] = sum([self.indicator(j, y_i) for y_i in y[0]])/self.k

        return probs

    # TODO: Implement the predict function to return the class for which it thinks is right
    def predict(self, x:np.ndarray, k:int=None) -> int:
        if k is not None:
            self.k = k
        neighbors =  self._generate_n_neighbors(x)
        class_probabilities = self._generate_probs(neighbors)
        print(class_probabilities.T)
        return np.argmax(class_probabilities)


if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]).T
    y = np.array([[0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]]).T

    knn = KNN()
    knn.fit(X, y)
    print(knn.predict(np.array([[3]]), k=7))