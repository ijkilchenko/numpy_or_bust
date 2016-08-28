import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from bokeh.charts import Scatter, show


class LogisticRegression:
    def __init__(self):
        self.G = np.vectorize(self.g)
        self.log = np.vectorize(self.log)

    def fit(self, X, y, num_iter=100, alpha=0.01):
        """Train the model on (X, y).
        X is a pandas DataFrame and y is a pandas Series (or a list). """
        self.K = len(set(y))  # Number of classes.
        # Dimension of the examples and number of examples.
        self.n, self.d = X.shape
        # Weights of Logistic Regression (plus bias).
        self.W = np.zeros((self.d + 1, self.K))

        # Add the biased feature.
        ones_column = pd.Series(np.ones(self.n))
        self.X = X.copy()
        self.X.reset_index(drop=True, inplace=True)
        self.X['bias'] = ones_column  # n by d+1 DataFrame.

        self.Y = pd.get_dummies(y)  # n by k DataFrame.

        # Gradient descent.
        for i in range(num_iter):
            if i % 10 == 0:
                # Print the actual loss every 10 iterations.
                print('Iteration: %i. Loss: %0.2f. Accuracy: %0.2f' %
                      (i, self._calc_Loss(), self.score(self.X, y)))
            dLossdw = self._calc_dLossdW()
            self.W -= alpha * dLossdw

    @staticmethod
    def g(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def log(x):
        return np.log(x)

    def score(self, X, y):
        y_predictions = self.predict(X)
        is_correct = y_predictions == y
        num_correct = len([i for i in is_correct if i])
        return num_correct/len(y)

    def predict(self, X):
        H = self.predict_proba(X)
        y = []
        for row in range(H.shape[0]):
            proba = H[row,:].tolist()
            y.append(proba.index(max(proba)))
        return y

    def predict_proba(self, X):
        ones_column = pd.Series(np.ones(X.shape[0]))
        X = X.copy()
        X.reset_index(drop=True, inplace=True)
        X['bias'] = ones_column  # n by d+1 DataFrame.

        Z = X.dot(self.W)
        H = self.G(Z)
        return H

    def _calc_Loss(self):
        """Calculate the log-loss. """
        Z = self.X.dot(self.W)
        # n by k DataFrame of predictions for each example and class.
        H = self.G(Z)

        # Vectorized log-loss over all examples and classes.
        loss = -(np.multiply(self.log(H), self.Y) +
                 np.multiply(self.log(1-H), 1-self.Y))
        return loss.sum().sum()

    def _calc_dLossdW(self):
        Z = self.X.dot(self.W)
        H = self.G(Z)

        # To be the matrix of partial derivatives (same shape as W).
        dLossdw = np.zeros(self.W.shape)

        dLoss = -(np.multiply(1-H, self.Y) + np.multiply(-H, 1-self.Y))
        # TODO: Vectorize the following calculation over classes.
        for k in range(self.K):
            grad_k = np.multiply(dLoss.iloc[:,k].reshape((1, dLoss.shape[0])),
                                 self.X.T).T
            grad_k = grad_k.sum(axis=0)
            grad_k = grad_k / self.n
            dLossdw[:,k] = grad_k
        return dLossdw

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000,
                               n_features=2, n_redundant=0, n_repeated=0,
                               n_classes=4, n_clusters_per_class=1,
                               random_state=43)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print('Accuracy after training (train): %0.2f' %
          model.score(X_train, y_train))
    print('Accuracy after training (test): %0.2f' %
          model.score(X_test, y_test))

    df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
    df.columns = ['x', 'y', 'cluster']
    p = Scatter(df, x='x', y='y', color='cluster')

    show(p)
