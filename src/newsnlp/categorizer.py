import torch

from transformers import AutoModelForSequenceClassification, TextClassificationPipeline

from newsnlp.base import Pretrained


class Categorizer(Pretrained):
    """

    https://huggingface.co/lincoln/flaubert-mlsum-topic-classification


    Categories:
        - fr: [ "Culture", "Economie", "Education", "Environement", "Justice",
                "Opinion", "Politique", "Societe", "Sport", "Technologie" ]
    """

    config = {
        "fr": {
            "model": "lincoln/flaubert-mlsum-topic-classification",
            "tokenizer": "lincoln/flaubert-mlsum-topic-classification"
        }
    }

    def __init__(self, lang, **kwargs):
        self.model, self.tokenizer = self.load(lang, AutoModelForSequenceClassification, **kwargs)
        self.nlp = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)

    def __call__(self, text):
        return self.categorize(text)

    def categorize(self, text, truncation=True):
        """
        :returns [(label, score), ...]
        """
        classes = self.nlp(text, truncation=truncation)
        return [(c['label'], c['score']) for c in classes]






