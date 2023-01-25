import os
from os.path import dirname, realpath

from transformers import AutoTokenizer, AutoConfig
from .utils.helpers import get_env_variable, run


DATA_DIR = get_env_variable('DATA_DIR', f"{dirname(realpath(__file__))}/data")
CACHE_DIR = get_env_variable('CACHE_DIR', None) # use default cache setting.


class Pretrained:
    """
    Loads pretrained model and companion tokenizer. Two workflows: online (default) or offline.
    Online: attempts to read data from cache: downloads the data if cache miss.
    Offline: attempts to load data from DATA_DIR; failover to downloading to DATA_DIR.
    Configurable DATA_DIR and CACHE_DIR as env variables.
    """

    config = None

    def load(self, lang, model_class, offline=False):
        """
        Loads tokenizer and model from config, fails over to HuggingFace's repo.
        :param model_class: eg. `transformers.AutoModelForSeq2SeqLM`, etc.
        :param str lang: language code, eg. `fr`, `en` to pick from `.config`.
            Must exist as a key in this `.config` class attribute!
        :param boolean offline: should cache tokenizer and model locally,
            ie, to the `./data/` subdir? further calls will first hit cache.
        """
        model, tokenizer = None, None

        def _from_cache(**kwargs):
            """ Returms model and tokenizer from cache.
            Attempts to download them to cache iff cache miss.
            """
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
            model = model_class.from_pretrained(model_name, **kwargs)
            return model, tokenizer

        # assert required params
        assert self.config, "`ConfigLoader.config` required."
        assert lang in list(self.config), f"language not supported: {lang}"
        model_name = self.config[lang]["model"]
        tokenizer_name = self.config[lang]["tokenizer"]

        if not offline:
            model, tokenizer = _from_cache()

        else:
            try:
                print(run(f"ls -lhrat {DATA_DIR}/{tokenizer_name}"))
                tokenizer = AutoTokenizer.from_pretrained(f"{DATA_DIR}/{tokenizer_name}", cache_dir=CACHE_DIR)
                model = model_class.from_pretrained(f"{DATA_DIR}/{model_name}", cache_dir=CACHE_DIR)

            except Exception as e: # really, shoud catch: ValueError, HFValidationError
                print(f"cannot find pretrained in DATA_DIR, attempting download ...\n"
                      f"->{DATA_DIR}/{tokenizer_name}\n"
                      f"->{DATA_DIR}/{model_name}")

                # first, download to cache, then, save pretrained to DATA_DIR for subsequent offline calls
                # nota: AutoTokenizer requires a config file, even if the tokenizer was found in the path??
                # https://github.com/huggingface/transformers/issues/6368#issuecomment-671250169
                # https://github.com/huggingface/transformers/issues/4197#issuecomment-625496433
                model, tokenizer = self._from_cache()
                tokenizer.save_pretrained(f"{DATA_DIR}/{tokenizer_name}")
                tokenizer_config.save_pretrained(f"{DATA_DIR}/{tokenizer_name}")
                model.save_pretrained(f"{DATA_DIR}/{model_name}")

        return model, tokenizer


