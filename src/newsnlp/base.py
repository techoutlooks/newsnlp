from os.path import dirname, realpath

from transformers import AutoTokenizer
from .utils.helpers import get_env_variable


NLP_DATA_DIR = get_env_variable('NLP_DATA_DIR', f"{dirname(realpath(__file__))}/data")


class ModelLoader:

    config = None

    def load(self, lang, model_class, cache=True):
        """
        Loads tokenizer and model from config, fails over to HuggingFace's repo.
        :param model_class: eg. `transformers.AutoModelForSeq2SeqLM`, etc.
        :param str lang: language code, eg. `fr`, `en` to pick from `.config`.
            Must exist as a key in this `.config` class attribute!
        :param boolean cache: should cache tokenizer and model locally,
            ie, to the `./data/` subdir? further calls will first hit cache.
        """

        assert self.config, "`ConfigLoader.config` required."
        assert lang in list(self.config), f"language not supported: {lang}"

        model_name = self.config[lang]["model"]
        tokenizer_name = self.config[lang]["tokenizer"]

        try:                # search cache first
            tokenizer = AutoTokenizer.from_pretrained(f"{DATA_DIR}/{tokenizer_name}")
            model = model_class.from_pretrained(f"{DATA_DIR}/{model_name}")

        except ValueError as e:  # download iff cache miss
            print("cannot find the requested files in the cached path, attempting download ...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = model_class.from_pretrained(model_name)

            if cache:        # cache all to ./data subdir
                tokenizer.save_pretrained(f"{DATA_DIR}/{tokenizer_name}")
                model.save_pretrained(f"{DATA_DIR}/{model_name}")

        return model, tokenizer
