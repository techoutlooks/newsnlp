from os.path import dirname, realpath

from transformers import AutoTokenizer


class ConfigLoader:

    config = None

    def load_from_config(self, lang, model_class):

        assert self.config, "`ConfigLoader.config` required."
        assert lang in list(self.config), f"language not supported: {lang}"

        model_name = self.config[lang]["model"]
        tokenizer_name = self.config[lang]["tokenizer"]

        src = dirname(realpath(__file__))
        tokenizer = AutoTokenizer.from_pretrained(f"{src}/data/{tokenizer_name}")
        model = model_class.from_pretrained(f"{src}/data/{model_name}")

        return model, tokenizer
