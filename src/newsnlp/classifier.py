import pandas as pd
import torch.nn as nn
import torch

from newsnlp.estimators import get_categorical_counts
from newsnlp.utils.helpers import load_fixture, save_fixture


class DeepClassifier(nn.Module):
    """
    Deep encoding of inputs categorical and continuous inputs of variable dimensions.
    Adapted from : https://jonnylaw.rocks/posts/2021-08-04-entity-embeddings/

    Following layers:
    - entity embeddings of categorical variables into a single (flattened) categorical predictor
    - distinct linear layer for continuous variables
    - single ReLU activation

    Keyword Arguments:

    num_output_classes: int
    num_cat_classes: list[int] number of classes by category
    num_cont: int
    embedding_dim: int
    hidden_dim: int

    TODO: encoding pure binary data (`local` input) as dummy to reduce complexity?
        no need to learn embeddings for mere 0 and 1 states
        unless we anticipate more inter-related states ...
    """

    def __init__(self, num_output_classes, cat_dims, num_cont, hidden_dim=50):

        super().__init__()

        # Create one embedding layer for each categorical input
        # +2 to assign missing (NaN) and unknown classes to a category of their own
        # embed_dims = 4th root of the number of classes in a given category (by Google)
        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        embed_dims = [int(nc ** .25) for nc in cat_dims]
        self.embeddings = nn.ModuleList([
            nn.Embedding(nc+2, embed_dim) for nc, embed_dim in zip(cat_dims, embed_dims)])

        #
        fc1_out_dim = int((sum(embed_dims)+num_output_classes) / 2)
        self.fc1 = nn.Linear(in_features=sum(embed_dims), out_features=fc1_out_dim)
        fc2_out_dim = int((num_cont + num_output_classes) / 2)
        self.fc2 = nn.Linear(in_features=num_cont, out_features=fc2_out_dim)

        self.relu = nn.ReLU()
        self.out = nn.Linear(fc1_out_dim+fc2_out_dim, num_output_classes)

    def forward(self, x_cat, x_con):
        # Embed each of the categorical variables
        x_embed = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(x_embed, dim=1)
        x_embed = self.fc1(x_embed)
        x_con = self.fc2(x_con)
        # x = torch.cat([x_con, x_embed.squeeze()], dim=1)
        x = torch.cat([x_con, x_embed], dim=1)

        x = self.relu(x)
        return self.out(x)

    @classmethod
    def from_data(cls, df, cat_cols, num_classes, save_to=None, **kwargs):
        """
        Create classifier instance with weights calibrated after the size of the data
        :param pd.DataFrame df: data to infer the weights dimensions from
        :param [str] cat_cols: identifies the categorical columns. assumes all other columns continuous.
        :param int num_classes: number of output classes to predict
        :param str save_to: path where to save data counts
        """
        cat_counts = get_categorical_counts(df, cat_cols)
        X_con = df.drop(cat_cols, axis=1)

        cont_dim, cat_dims = len(X_con.columns), \
            [len(count) for count in cat_counts.values()]

        if save_to:
            save_fixture(save_to, {
                'categorical': cat_counts,
                'continuous': X_con.describe().to_dict()})

        return cls(num_classes, cat_dims, cont_dim, **kwargs)

    @classmethod
    def from_json(cls, path, cat_cols, num_classes, **kwargs):
        stats = load_fixture(path)
        cont_dim = len(stats['continuous'])
        cat_dims = [len(v) for k, v in stats['categorical'].items() if k in cat_cols]
        return cls(num_classes, cat_dims, cont_dim, **kwargs)


def create_classifier(dims, cat_cols, num_classes, **kwargs):
    """
    Initialize the deep classifier from versatile dimensions sources
    Dimensions refer to the embeddings input sizes (one per categorical variable),
    and the number of continuous predictors.

    :param str or pd.DataFrame dims: actual dimensions, or a source to infer model dimensions from,
    ie. either: train set dataframe, or previously saved fixture file (JSON).
    If inputted the dims manually, careful not to raise out-of-bounds IndexError!

    :param int num_classes: number of output classes to predict
    :param [str] cat_cols: labels of the categorical columns
    """

    if isinstance(dims, (tuple, list)) and len(dims) == 2:
        cat_counts, num_cont = dims
        return DeepClassifier(num_classes, cat_counts, num_cont, **kwargs)

    elif isinstance(dims, pd.DataFrame):
        return DeepClassifier.from_data(dims, cat_cols, num_classes, **kwargs)

    elif isinstance(dims, str):
        return DeepClassifier.from_json(dims, cat_cols, num_classes, **kwargs)

