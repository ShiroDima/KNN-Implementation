# K-Nearest Neighbor model for Classification and Regression.

## Classification
Given a positive integer K and a test observation $x_0$, the classifier first creates a set $N_0$ that contains the 
K points in the training set that are closest to $x_0$.
The `KNN model` for classification tries to estimate the conditional probability for class j as the fraction of points in $N_0$ whose response values equal j:
$$Pr(Y=j|X=x_0) = \frac{1}{K} \sum_{i \in N_0}{I(y_i = j)}$$

The KNN classifies the test observation $x_0$ to the class with the highest probability.

## Regression
Given a positive integer K and a test observation $x_0$, the classifier first creates a set $N_0$ that contains the 
K points in the training set that are closest to $x_0$.
The `KNN model` for regression returns as its prediction, the average of all the y values for each $x_i$ in $N_0$:
$$ \hat{f}(x_0) = \frac{1}{K} \sum_{x_i \in N_0}{y_i}$$