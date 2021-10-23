"""
LSTM

    Legend:
        - m=vocab_size (#inputs), n=batch_size (#samples)
        - X (n, m): word vectors, from sentence i
        - H (n, n_h) : hidden state at time step (t)
        - C (n, n_h) : hidden memory cell
        - state = (H, C) : LSTM's hidden state

    Hyper-params:
        - n_h : number of neurons if the hidden layer
        - n_i : number of inputs
        - n_o : number of outputs
        - µ, σ of normal distribution: for weights initialization
        - batchsize : >1 for mini-batch gradient descent (eg. 32?)
        - timesteps : number of time steps. one per mini-batch; (eg. 1000?)
        - epochs : number times we'll loop through the entire training dataset.

    Language models:
        n_i = n_o = m, m-dimensional word vectors.

"""
import autograd.numpy as np


GAUSSIAN_MU = 0
GAUSSIAN_SIGMA = 0.1


class RNNModel:

    def __init__(self, vocab_size, n_h):
        pass


class Lstm:
    """
    LSTM network.
    """

    def __init__(self, vocab_size, n_h):
        pass

    def get_params(self, vocab_size, n_h, n_o, mu=GAUSSIAN_MU, sigma=GAUSSIAN_SIGMA):

        normal = lambda shape, mu, sigma: np.random.normal(loc=mu, scale=sigma, size=shape)

        def three():
            """
            Initializes the weights (Gaussian) and biases (to zero)
            of (resp) the: forget, input, output gates, memory cell, and predictor

                W_xf, W_xi, W_xo, W_xc  : (m, n_h)    - weights of inputs (eg. word vectors' components)
                W_hf, W_hi, W_ho, W_hc  : (n_h, n_h)  - weights of hidden state of previous time step
                W_hy                    : (h, n_o)    - predictor's weights

                b_f, b_h, b_o, b_c      : (n,1)       - biases of candidate cell
                b_y                     : (h,1)       - predictor's bias

            :returns
                (
                    W_xf, W_hf, b_f,    # forget gate params
                    W_xi, W_hi, b_i,    # input get params
                    W_xo, W_ho, b_o,    # output gate params
                    W_xc, W_hc, b_c,    # candidate memory cell params
                    W_hy, b_h           # predictor's params
                )
            """
            return (
                normal((vocab_size, n_h), mu, sigma),
                normal((n_h, n_h), mu, sigma),
                np.zeros(vocab_size)
            )

        params = [
            three(), three(), three(), three(),
            normal((n_h, n_o), mu, sigma), np.zeros(n_h)
        ]

        grads = [np.zeros_like(p) for p in params]

        return params, grads

    def forward(self, inputs, state, params):
        """
        Forward propagation.
        Computes the current state (timestep t) and the predictor Y.
            C<t> = C<t-1> * F  + C_tilda * I, previous knowledge + current analysis
            H = tanh(C<t>) * O, a counter really
            Y = H.T x W_hy + b_y, no activation function

        :params inputs: (`num_steps`, `batch_size`, `vocab_size`)
        :param state: state at t-1
        :param params:
        :return:
        """

        # previous states (t-1, which contains t-2, t-3, ... recursively)
        H, C = state

        W_xf, W_hf, b_f, \
        W_xi, W_hi, b_i, \
        W_xo, W_ho, b_o, \
        W_xc, W_hc, b_c,\
        W_hy, b_y \
            = params

        # Shape of `X`: (`batch_size`, `vocab_size`)
        outputs = []
        for X in inputs:
            F = self.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
            I = self.sigmoid(np.tanh(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i))
            O = self.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
            C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)

            # current state (t)
            C = F * C + I * C_tilda
            H = O * np.tanh(C)

            Y = np.dot(H, W_hy) + b_y
            outputs.append(Y)

        # predictor value
        Y = np.concatenate(Y, axis=0)

        return (H, C), Y

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        """
        Performs multidimensional softmax.
        1D softmax transforms a vector of logit values `x` into another where:
        (a) output logits are probabilities, ie., positive values, summing up to 1
        (b) outputs as a function is differentiable.

        https://stackoverflow.com/a/59111948
        """
        pass

