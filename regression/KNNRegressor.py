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
        try:
            n_cols = x.shape[1]
        except IndexError:
            raise IndexError("If predicting on a single sample, use x.reshape(1, -1)")

        if self.X.shape[1] != n_cols:
            raise ValueError("input x must have the same number of columns as the data matrix")

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


    def _find_average(self, y:np.ndarray) -> np.array:
        return np.sum(y, axis=1) / self.k


    def predict(self, x:np.ndarray, k:int=None) -> int:
        if self.X is None or self.y is None:
            raise TypeError("Please fit the model before trying to use it for predictions")
        if k is not None:
            self.k = k
        neighbors =  self._generate_k_neighbors(x)
        prediction = self._find_average(neighbors)
        return prediction


if __name__ == '__main__':
    reg_data = pd.read_csv("../data/cal_house_pricing.csv")
    X, y = reg_data.drop(columns=["y"]), reg_data[["y"]]
    X_pred_reg = X.iloc[:5].values
    knn = KNNRegressor()
    knn.fit(X, y)
    print(knn.predict(X_pred_reg, k=5))
