import abc

import numpy as np


class Adagrad:

    def __init__(self, W, epsilon, eta=1e-8,):
        self.W = W
        self.eta = eta
        self.epsilon = epsilon

    def forward(self):
        pass


def sgd(params, grads, state, hyper_params):

    for p, g in zip(params, grads):
        p[:] -= hyper_params["lr"] * g

    return params


class Model(abc.ABC):

    @property
    @abc.abstractmethod
    def params(self):
        pass

    @property
    @abc.abstractmethod
    def grads(self):
        pass

    def train(self, solver, hyper_params, X_train, y_train):

        for epoch in range(hyper_params["epochs"]):

            for X, y in self.data_iter(X_train, y_train, hyper_params["batch_size"]):
                self(X, y)
                solver(self.params, self.grads, None, hyper_params)
                pass

            print(f'epoch {epoch}, loss={self.l2_loss()}')
        pass

    def data_iter(self, features, labels, batch_size, shuffle=False):
        """
        Yields minibatch (X, y) of size batch_size
        from a dataset of (features, labels).
        """
        m = len(labels)
        indices = list(range(m))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, m, batch_size):
            batch_indices = np.array(indices[i: min(i+batch_size, m)])
            yield features[batch_indices], labels[batch_indices]


class LinRegModel(Model):
    """
    Linear regression, one neuron at a time.
    http://vxy10.github.io/2016/06/25/lin-reg-matrix/
    """

    def __init__(self, num_features, batch_size):
        self.n = num_features
        self.m = batch_size
        self.w, self.b = \
            np.zeros((self.n, 1)), \
            np.zeros((self.m, 1))

    @property
    def params(self):
        return self.w, self.b

    def __call__(self, X, y):
        self.X, self.y = X, y
        self.y_hat = np.zeros_like(y)
        return self

    def forward(self):
        return np.dot(self.X, self.w) + self.b

    @property
    def grads(self):
        """
        Computes gradients analytically, evaluate them at X
            f = (Xw - y)'(Xw - y)/2
            𝝏f/𝝏b = Xw + b - y
            𝝏f/𝝏w = X'Xw - X'y + X'b) = X'Xw - X'y + X'b = X'(Xw +b - y)
            𝝏f/𝝏w = X'𝝏f/𝝏b
        """

        db = np.dot(self.X, self.w) + self.b - self.y
        dw = np.dot(self.X.T, db)

        return dw, db

    def l2_loss(self):
        """ MSE loss, an entire layer at once. """
        return 0.5 * (self.y_hat - self.y) ** 2


EPOCHS = 5
BATCH_SIZE = 50
LEARNING_RATE = 0.1


def fake_data(w, b, num_samples):
    """ 𝐲=𝐗𝐰+𝑏+𝜖 """
    X = np.random.normal(0, 1, (num_samples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, .01, y.shape)

    return X, y.reshape(y.shape[0], 1)


if __name__ == '__main__':

    # dataset = np.genfromtxt()
    true_w, true_b = np.array([2, -.4]), 4.2
    X, y = fake_data(true_w, true_b, 1000)
    m, n = X.shape

    # train/test split by features (X) and labels (y)
    # X_train, y_train, X_test, y_test = \
    #     skip_gram[:m*3//4, :-1], skip_gram[:m*3//4, -1:], \
    #     skip_gram[m*3//4:, :-1], skip_gram[m*3//4:, -1:]
    X_train, y_train, X_test, y_test = \
        X[:m*3//4], y[:m*3//4], \
        X[m*3//4:], y[m*3//4:]

    # hyperparams
    _, n = X_train.shape
    hyper_params = dict(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_features=n,
        lr=LEARNING_RATE,
    )

    # define & train model
    nn = LinRegModel(hyper_params['num_features'], hyper_params['batch_size'])
    nn.train(sgd, hyper_params, X_train, y_train)


