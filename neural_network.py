from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from itertools import product
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation


class NeuralNetwork:
    g = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))
    log = np.vectorize(lambda x: np.log(x))

    def __init__(self, test=False):
        self.test = test  # Are we in test mode?
        # Vectorize activation and log functions.

    def fit(self, X, y, architecture, num_iter=100, alpha=0.01, animate=False):
        """architecture: defines the structure of our neural network
        and also encodes the number of classes.

        num_iter: number of iterations of gradient descent.
        alpha: gradient descent learning rate.
        """

        self.L = len(architecture)+1
        self.K = architecture[-1]
        self.d = X.shape[1]
        self.arch = [self.d, *architecture]
        self.Ws = []
        for i in range(self.L - 1):
            if self.test:
                np.random.seed(2016)
                # +1 for the bias neuron.
                W = np.random.rand(self.arch[i] + 1, self.arch[i + 1])
            else:
                np.random.seed()
                # +1 for the bias neuron.
                W = np.random.rand(self.arch[i] + 1, self.arch[i + 1])
            self.Ws.append(W)

        self.n = X.shape[0]
        self.all_ones = pd.Series([1]*self.n)
        X = X.copy()  # Do not modify the data set.
        X.reset_index(drop=True, inplace=True)
        self.X = pd.concat([self.all_ones, X], axis=1)
        self.Y = pd.get_dummies(y).reset_index(drop=True)

        if animate:
            grids = []  # For the animation of the regions.
            # grid_x = np.linspace(X[0].min(), X[1].max(), 50)
            # grid_y = np.linspace(X[0].min(), X[1].max(), 50)
            grid_x = np.linspace(-3, 3, 50)
            grid_y = np.linspace(-3, 3, 50)
            grid = pd.DataFrame(list(product(grid_x, grid_y)))
            grid.columns = ['x', 'y']
            accuracies = []

        # Gradient descent.
        if self.test:
            num_iter = 2
        for i in range(num_iter):
            if not self.test and i % 10 == 0:
                acc = self.score(X, y)
                # Print the actual loss every 10 iterations.
                print('Iteration: %i. Loss: %0.2f. Accuracy: %0.5f' %
                      (i, self._calc_loss(self.Ws), acc))
                if animate:
                    grid_predictions = self.predict(grid)
                    #grid_predictions = np.random.random(grid.shape[0])
                    grids.append(grid_predictions)
                    accuracies.append(acc)
            jacs = self._calc_dLoss_dWs()  # Matrix of partial derivatives.
            if self.test:
                # Numerically approximate the partial derivatives.
                jacs1 = self._num_calc_dLoss_dWs()
                for l in range(len(jacs)):
                    assert np.linalg.norm(jacs[l] - jacs1[l]) < 1e-3
            for j, jac in enumerate(jacs):
                self.Ws[j] -= alpha * jac  # Update rule.

        if animate:
            return grid, grids, accuracies

    def _calc_loss(self, Ws):
        # Extract the last set of activations (i.e. hypotheses)
        A = self.forward_pass(self.X, Ws)[0][-1]
        # Compute cost function (log loss).
        J = -(np.multiply(self.Log(1-A), 1-self.Y) +
              np.multiply(self.Log(A), self.Y))
        loss = J.sum().sum()
        return loss

    def score(self, X, y):
        y_predictions = self.predict(X)
        is_correct = [y_pred == y_true for y_pred, y_true in
                      zip(y_predictions, list(y))]
        num_correct = len([i for i in is_correct if i])
        return num_correct/len(y)

    def predict(self, X):
        H = self.predict_proba(X)
        y = []
        for row in range(H.shape[0]):
            proba = H.iloc[row,:].tolist()
            y.append(proba.index(max(proba)))
        return y

    def predict_proba(self, X):
        all_ones = pd.Series([1]*X.shape[0])
        X = X.copy()  # Do not modify the data set.
        X.reset_index(drop=True, inplace=True)
        X = pd.concat([all_ones, X], axis=1)

        H = self.forward_pass(X, self.Ws)[0][-1]
        return H

    def forward_pass(self, X, Ws):  # Compute all activations for each layer
        As = []  # Activations.
        Zs = []  # Values before the activation function is applied.
        A = X
        As.append(A)
        for i in range(self.L - 1):
            Z = np.dot(A, Ws[i])
            A = pd.DataFrame(self.G(Z))
            all_ones = pd.Series([1] * A.shape[0])
            if i != self.L - 2:  # Add bias units for all except the last layer.
                A = pd.concat([all_ones, A], axis=1)
            As.append(A)
            Zs.append(Z)

        return As, Zs

    def _num_calc_dLoss_dWs(self):  # Gradient checking.
        jacs = []
        for l in range(self.L-1):
            # Sum derivatives for each example.
            dx = 1e-5
            jac = np.zeros(self.Ws[l].shape)
            for i in range(self.Ws[l].shape[0]):
                for j in range(self.Ws[l].shape[1]):
                    loss = self._calc_loss(self.Ws)
                    Ws1 = [W.copy() for W in self.Ws]
                    Ws1[l][i,j] += dx
                    loss1 = self._calc_loss(Ws1)
                    rise = loss1 - loss
                    jac[i,j] = rise/dx
            jacs.append(jac)
        return jacs

    def _calc_dLoss_dWs(self): # Compute all partial derivatives for each layer.
        Ds = []  # Error terms (one for each z).
        As, Zs = self.forward_pass(self.X, self.Ws)
        D = As[-1] - self.Y  # Errors in the last layer.
        Ds.append(D)
        for i in reversed(range(1, self.L-1)):  # Backpropagate.
            # Expression for the Error term for all but the last layer.
            D = np.multiply(np.dot(Ds[-1], self.Ws[i].T)[:,1:],
                            np.multiply(self.G(Zs[i-1]), 1-self.G(Zs[i-1])))
            Ds.append(pd.DataFrame(D))
        Ds = Ds[::-1]  # Reverse the list (since we were appending).
        jacs = []  # Jacobian matrixes (matrixes of partial derivatives).
        for i in range(self.L-1):
            # Sum derivatives for each example.
            jac = np.zeros(self.Ws[i].shape)
            # TODO: Vectorize over the examples.
            for j in range(self.n):
                activations_col = As[i].iloc[j,:].T.reshape((As[i].shape[1],1))
                errors_row = Ds[i].iloc[j,:].reshape((1,Ds[i].shape[1]))
                # Partial derivatives for this example and this layer.
                outer_prod = np.dot(activations_col, errors_row)
                jac += outer_prod
            jacs.append(jac)
        return jacs

    @classmethod
    def G(cls, z):
        return cls.g(z)

    @classmethod
    def Log(cls, x):
        return cls.log(x)


def test():
    # Test the NeuralNetwork with gradient checking.
    X, y = make_classification(n_samples=50,
                               n_features=2, n_redundant=0, n_repeated=0,
                               n_classes=2, n_clusters_per_class=1,
                               random_state=206)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = NeuralNetwork(test=True)
    model.fit(X, y, [3, 2])

if __name__ == '__main__':
    test()  # Do gradient checking (did we write down the derivative correctly?)

    X, y = make_classification(n_samples=50,
                               n_features=2, n_redundant=0, n_repeated=0,
                               n_classes=2, n_clusters_per_class=2,
                               random_state=2019)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)

    model = NeuralNetwork()
    grid, grids, accuracies = model.fit(X_train, y_train,
                                        architecture=[3, 2],
                                        animate=True)
    print('Accuracy after training (train): %0.2f' %
          model.score(X_train, y_train))
    print('Accuracy after training (test): %0.2f' %
          model.score(X_test, y_test))

    def init_animation():
        global text
        text = ax.text(0.1, 0.1, '', transform=ax.transAxes)
        global region
        region = ax.scatter(grid['x'], grid['y'], alpha=0.4, lw=0,
                            animated=True)

    def animate(i):
        text.set_text('Accuracy: %0.5f' % accuracies[i])
        region.set_array(np.array(grids[i]))

    df = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)
    df.columns = ['x', 'y', 'cluster']

    fig = plt.figure()

    plt.scatter(df['x'], df['y'], c=df['cluster'], alpha=1, lw=0)
    ax = fig.add_subplot(111)

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                                             init_func=init_animation,
                                             frames=len(accuracies))
    ani.save('./animation.gif', writer='imagemagick', fps=3)
