import pandas as pd
import spacy
from sklearn.base import TransformerMixin, BaseEstimator


__all__ = ("CategorySelector",)


class CategorySelector(BaseEstimator, TransformerMixin):
    """
    Of the classes of given categories, retain only significant possibilities,
    in a attempt to reduce datasets and convergence time of Neural Nets.
    Significance is assessed using the TextRank algorithm.
    """

    def __init__(self, lang, model, cat_cols):
        self.nlp = spacy.load(lang)
        self.model = model
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
            truncate instances with more classes than embeddings (mapping must be bijective)
        however, we anticipate training to see more classes than avail in instances at inference
        how to fairly decide which extra classes to delete? recourse to some term
        it is wise to delete incriminated instances altogether?
        experiment with merely truncating the extra dimensions from the category

        :param pd.DataFrame X: source data
        :param model: instantiated model to get the embed_dims of reference from

         TODO: as ColumnTransformer -> `data_loader.preprocess_data()`
        """

        # examine the top-ranked phrases in the document
        # add PyTextRank to the spaCy pipeline
        self.nlp.add_pipe("text_rank")

        for i, cat in enumerate(self.cat_cols):
            num_embeds = self.model.embeddings[i].num_embeddings
            X.loc[:, cat] = X.apply(lambda x: self.extract_classes(
                x[cat], x._raw[cat], num_embeds), axis=1)

        return X

    def extract_classes(self, cat_classes: list, cat_raw_text: str, limit: int) -> pd.Series:
        """

        """

        has_extra_classes = lambda x_cat, limit: \
            isinstance(x_cat, (list, tuple)) and len(x_cat) > limit

        if has_extra_classes(cat_classes):
            doc = self.nlp(cat_raw_text)
            for phrase in doc._.phrases[:limit]:
                cat_classes = list(set(cat_classes))[:limit]

        return cat_classes

