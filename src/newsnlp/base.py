from os.path import dirname, realpath

from transformers import AutoTokenizer, AutoConfig
from .utils.helpers import get_env_variable


NLP_DATA_DIR = get_env_variable('NLP_DATA_DIR', f"{dirname(realpath(__file__))}/data")


class ModelLoader:
    """
    Supplies `NLP_DATA_DIR` env, for custom location to save the pretrained models.
    """

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

        # first, search NLP_DATA_DIR
        try:
            tokenizer = AutoTokenizer.from_pretrained(f"{NLP_DATA_DIR}/{tokenizer_name}")
            model = model_class.from_pretrained(f"{NLP_DATA_DIR}/{model_name}")

        # cache miss: download pretrained -> NLP_DATA_DIR
        # really, shoud catch: ValueError, HFValidationError
        # nota: AutoTokenizer requires a config file, even if the tokenizer was found in the path
        except Exception as e:
            print(f"cannot find pretrained in NLP_DATA_DIR, attempting download ...\n"
                  f"->{NLP_DATA_DIR}/{tokenizer_name}\n"
                  f"->{NLP_DATA_DIR}/{model_name}")

            # first, download to cache
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer_config = AutoConfig.from_pretrained(tokenizer_name)
            model = model_class.from_pretrained(model_name)

            # then, save pretrained to NLP_DATA_DIR for future calls.
            # https://github.com/huggingface/transformers/issues/6368#issuecomment-671250169
            if cache:
                tokenizer.save_pretrained(f"{NLP_DATA_DIR}/{tokenizer_name}")
                tokenizer_config.save_pretrained(f"{NLP_DATA_DIR}/{tokenizer_name}")
                model.save_pretrained(f"{NLP_DATA_DIR}/{model_name}")

        return model, tokenizer
