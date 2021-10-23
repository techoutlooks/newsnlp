import torch
from matplotlib import pyplot as plt
from torch import nn

from src.newsnlp.skip_gram.data_loader import DataLoader
from src.newsnlp.utils.vocab import Vocab


class SkipGram(nn.Module):
    """
    Learns dense embeddings inside matrix in_embed.

    "biases are not used in the neural network,  as no significant improvement
    of performance was observed" - Mikolov (2012)

    Inspired from: https://gist.github.com/GavinXing/9954ea846072e115bb07d9758892382c
    """

    def __init__(self, vocab_size, embed_dim):
        """

        :param context_size: size of context window
        :param embed_dim: dim of the learned word-vectors representations (lower dimensional space)
        :param vocab_size: |in_embed|, number of uniqute words in vocab
        """
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)             # performs XV,  learns V (skip-gram embeddings)
        self.out_embed = nn.Linear(embed_dim, vocab_size, bias=False)   # performs XVU, learns U

        nn.init.uniform(self.in_embed.weight, 0, 1)
        nn.init.uniform(self.out_embed.weight, 0, 1)

    def forward(self, X: [torch.LongTensor]):
        """
        Predict the context words (Y) given a center word (x),
        ie. one-hot encoded center words.

        :param X: single sentence, ie. minibatch of tokens (sample)
        :return: predicted context y_hat (labels)
        """
        embeds = self.in_embed(X)
        Z = self.out_embed(embeds)
        Y_hat = nn.functional.softmax(Z)
        return Y_hat, (Z,)

    def get_embedings(self):
        V = self.in_embed.weight
        U = self.out_embed.weight
        return V, U

    def fit(self, optimizer, loss_fn, hyperparams, data_iter):
        """
        Main training loop -
        Learn in_embed, out_embed using minibatch gradient descent,
        train on (X, y), X=center words, y=context words (labels).

        :param optimizer: SGD, ADAM, RMSProps, etc.
        :param loss_fn: eg. CrossEntropyLoss (CE)
        :param data_iter: iterable dataset of minibatches (X, y)
        :return:
        """
        loss_history = []
        for epoch in range(hyperparams["num_epochs"]):

            # (X, Y) pairs of indices of words (center, w_o)
            # from (center, context) for every w_o in context
            for X, Y in data_iter:
                Y_hat, (Z,) = self.forward(X)
                # loss = loss_fn(Z, torch.FloatTensor(data_iter.vocab.one_hot(Y)))
                loss = loss_fn(Z, Y)   # CE loss = -y * log(y_hat), per minibatch (center, context)
                loss.backward()                  # computes gradients wrt. in_embed.weight, out_embed.weight
                optimizer.step()                 # optimize weights using ADAM
                optimizer.zero_grad()

                # plotting purposes
            loss_history += [loss]

        # detach gradients, 3 ways:
        # >>> loss_history.detach().numpy()
        # >>> with torch.no_grad()
        # >>> [l.item() for l in loss_history]
        with torch.no_grad():
            plt.xlabel = "epoch"
            plt.ylabel = "loss"
            plt.title = f'CE Loss Profile - context_window_size={hyperparams["context_window_size"]}'
            plt.plot(range(hyperparams["num_epochs"]), loss_history)

        plt.show()

        pass


def get_data(context_window_size, max_tokens=None):
    with open("../test/data/le_loup_et_le_chien.txt", 'r') as f:
        raw_text = f.read()

    loader = DataLoader(context_window_size, max_tokens=max_tokens)
    loader(([raw_text]))
    return loader


def get_similar_words(word: str, top_k: int, model: SkipGram, vocab: Vocab, ):
    word_ix = torch.LongTensor([vocab[word]])
    Y_hat, _ = model.forward(word_ix)
    results = torch.argsort(Y_hat, descending=True)

    # take indexes of predictions, from highest to lowest
    # top_k values along default dim (=0 dim)
    return [vocab[ix.item()] for ix in results[-1, :top_k]]


if __name__ == '__main__':

    CONTEXT_WINDOW_SIZE = 5
    EMBED_DIM = 10              # most commonly 300
    NUM_EPOCHS = 1000

    # sense of CE loss with softmax?
    # https://jaketae.github.io/study/neural-net/
    def cross_entropy_loss(Z, Y):
        return -torch.sum(Y * torch.log(Z))


    data_iter = get_data(CONTEXT_WINDOW_SIZE)
    hyperparams = dict(

        # training
        num_epochs=NUM_EPOCHS,
        lr=.0001,

        # skip-gram
        feature_dim=len(data_iter.vocab),
        embed_dim=EMBED_DIM,
        context_window_size=CONTEXT_WINDOW_SIZE,
    )

    model = SkipGram(hyperparams["feature_dim"], hyperparams["embed_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    # loss_fn = cross_entropy_loss
    loss_fn = nn.CrossEntropyLoss()
    model.fit(optimizer, loss_fn, hyperparams, data_iter)

    # 10 top similar words to "Loup"
    print(get_similar_words("Loup", top_k=10, model=model, vocab=data_iter.vocab))



