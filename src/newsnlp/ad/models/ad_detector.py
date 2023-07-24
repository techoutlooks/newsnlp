import logging
import optparse

import pandas as pd

import skorch
import torch

from skorch import helper as skorch_helper

from newsnlp.ad.constants import ADS_DATASET_STATS, saved_pretrained, ADS_DATASET, MAX_PHRASE_LEN
from newsnlp.ad import create_ad, Ad
from newsnlp.ad.data_loader import DataLoader, \
    ad_predictors, preprocess_data, clean_data, load_preprocessor

import newsnlp.globals as g
from newsnlp.classifier import create_classifier


nlp = g.load_nlp(lang='fr')


# Number of hidden layers for continuous predictors
# https://stats.stackexchange.com/a/1097
hidden_dim = 1


def create_ad_detector(model, from_pretrained=False, max_epochs=50):
    """
    Ad detector Neural Net as a classifier instance

    with default loss functions (binary cross entropy in this case),
    train/validation split logic, console logging of loss, validation loss etc.

    :param skorch.NeuralNet model: module to create from scratch / pre-trained
    :param bool from_pretrained: create the NN from previously saved pre-trained data?
    :param int max_epochs: num t
    """
    opts = {} if not from_pretrained else dict(
        iterator_train__shuffle=True,
        max_epochs=max_epochs, verbose=True)
    net = skorch.NeuralNetBinaryClassifier(module=model, **opts)

    if from_pretrained:
        net.initialize()  # This is important!
        net.load_params(**saved_pretrained)

    return net


def predict_ads(ads_data: dict, lang='fr') -> dict:
    """
    Predict ad candidates labels using a pre-trained model
    """
    # instantiate pretrained Neural Net.
    # model instance is calibrated (weights) from train data
    model = create_classifier(ADS_DATASET_STATS, ad_predictors.categorical, 1, hidden_dim=hidden_dim)
    net = create_ad_detector(model, from_pretrained=True)

    # map ads_data -> dataframe array
    ad_candidates = pd.DataFrame(map(lambda ad: create_ad(**ad, lang=lang), ads_data))

    # data preparation. transforms columns
    # load preprocessor fit from training (same ordinal encoding table)
    # Nota: kwargs past `preprocessor` are passed to the preprocessor
    X, counts = clean_data(ad_candidates, min_phrase_count=MAX_PHRASE_LEN, unwind_cols=ad_predictors.categorical)
    X, _ = preprocess_data(X, cont_cols=ad_predictors.continuous, cat_cols=ad_predictors.categorical,
                           preprocessor=load_preprocessor, drop_cols=['is_ad'])

    # forward pass
    Xs = skorch_helper.SliceDict(
        x_cat=X[ad_predictors.categorical].to_numpy(dtype="long"),
        x_con=torch.tensor(X[ad_predictors.continuous].to_numpy(), dtype=torch.float))
    y_pred = net.predict(Xs)

    X.reset_index(names='ad_id', inplace=True)
    X[Ad.label] = y_pred

    # prediction for given ad is the more frequent label class (is_ad or not)
    # hence, return the first mode per ad
    ad_preds = X.groupby(['ad_id'])[Ad.label]\
        .apply(lambda g: g.mode()[0])\
        .reset_index()

    # patch predictions with initial ad data
    ad_preds = pd.concat([ad_candidates, ad_preds[Ad.label]], axis=1)
    return ad_preds.to_dict(orient='records')


def train(max_epochs=10, save_params=True):
    """
    Training loop
    Also save trained params to storage (weights, history, and optimizer)
    """

    loader = DataLoader(lang='fr')(ADS_DATASET)
    loader.preprocess()
    X_train, X_test, y_train, y_test = loader.data

    model = create_classifier(X_train, ad_predictors.categorical, 1, hidden_dim=hidden_dim,
                              save_to=ADS_DATASET_STATS)
    net = create_ad_detector(model, max_epochs=max_epochs)

    # To pass multiple arguments to the forward method of the Skorch model we must specify a SliceDict
    # such that Skorch can access the data and pass it to the module properly.
    Xs = skorch_helper.SliceDict(
        x_cat=X_train[ad_predictors.categorical].to_numpy(dtype="long"),
        x_con=torch.tensor(X_train[ad_predictors.continuous].to_numpy(), dtype=torch.float))

    # Train the weights of the neural network using back-propagation.
    net.fit(Xs, y=torch.tensor(y_train.values, dtype=torch.float))

    # Saves the fitted model to persistent storage
    if save_params:
        net.save_params(**saved_pretrained)

    # Assess performance on the test set
    # using the encodings learnt on the training set
    Xs_test = skorch_helper.SliceDict(
        x_cat=X_test[ad_predictors.categorical].to_numpy(dtype="long"),
        x_con=torch.tensor(X_test[ad_predictors.continuous].to_numpy(), dtype=torch.float))

    # Mean accuracy on test data and labels
    score = net.score(Xs_test, y_test)
    print("score: ", score)


def main(argv=None):

    prs = optparse.OptionParser(
        usage="%prog [-v|--verbose] [--json|--json-indent] <path0> [<pathN>]",
        description="Print metadata for the given image paths "
                    "(without image library bindings).")

    prs.add_option('-e', '--max-epochs', dest='max_epochs', action='store', type='int')
    prs.add_option('-p', '--save-params', dest='save_params', action='store')
    prs.add_option('-v', '--verbose', dest='verbose', action='store_true')
    prs.add_option('-q', '--quiet', dest='quiet', action='store_true')

    argv = list(argv) if argv is not None else sys.argv[1:]
    (opts, args) = prs.parse_args(args=argv)
    loglevel = logging.INFO

    if opts.verbose:
        loglevel = logging.DEBUG
    elif opts.quiet:
        loglevel = logging.ERROR

    logging.basicConfig(level=loglevel)
    log = logging.getLogger()
    log.debug('argv: %r', argv)
    log.debug('opts: %r', opts)
    log.debug('args: %r', args)

    EX_OK = 0
    EX_NOT_OK = 2

    train(save_params=opts.save_params, max_epochs=opts.max_epochs)

    return EX_OK


if __name__ == "__main__":

    import sys
    sys.exit(main(argv=sys.argv[1:]))

