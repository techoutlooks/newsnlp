import os
from os.path import dirname as up, realpath

from transformers import AutoTokenizer, AutoConfig
from .utils.helpers import get_env_variable, get_path_stats


__all__ = (
    "data_dir", "cache_dir",
    "mk_model_path", "load_model",
    "Pretrained"
)


# DATA_DIR: alternate mount directory to check first, before failing over to HF repo scheme.
# CACHE_DIR: sets the `cache_dir` kwarg of HF's `from_pretrained()`. Default: `~/.cache/huggingface/transformers/`
# Better keep a clean `DATA_DIR` separate from `CACHE_DIR` since HF performs manipulations on the `CACHE_DIR`.
data_dir = get_env_variable('DATA_DIR', f"{up(up(realpath(os.curdir)))}/data")
cache_dir = get_env_variable('CACHE_DIR', None) # use default cache setting.


def mk_model_path(name, online=False):
    """ Gets ready to download from HuggingFace repo if online is truthy.
    :param name: model id, incl. namespace if any; eg. dbmdz/bert-base-german-cased.
    :param online: if false, look for model inside `DATA_DIR`.
    """
    path = name if online \
        else f"{data_dir.rstrip('/')}/{name}"
    return path


def load_model(model_class, model_name, tokenizer_name, save=False, online=False, **kwargs):
    """
    Download model and tokenizer to cache.
    Resolution: cache (if not kwargs.force_download) -> data_dir -> HuggingFace
    Force-download from HF repos if online==True.

    :param str model_class: name of Tokenizer
    :param str model_name: model id of a pretrained model hosted on `DATA_DIR` or huggingface.co repo.
    :param str tokenizer_name: name of Tokenizer
    :param bool save: whether to also save downloaded/cached files -> data_dir
    :param bool online: whether to force-download from HF instead of `DATA_DIR`
    :param kwargs: extra params acceptable for `from_pretrained()`
    """

    opts = {'cache_dir': cache_dir, **kwargs}
    tokenizer = AutoTokenizer.from_pretrained(mk_model_path(tokenizer_name, online), **opts)
    model = model_class.from_pretrained(mk_model_path(model_name, online), **opts)

    if save:
        tokenizer.save_pretrained(f"{data_dir}/{tokenizer_name}")
        model.save_pretrained(f"{data_dir}/{model_name}")

    return model, tokenizer


class Pretrained:
    """
    Loads pretrained model and companion tokenizer. Two workflows: online (default) or offline.
    Online: attempts to read data from cache: downloads the data if cache miss.
    Offline: attempts to load data from DATA_DIR; fail over to downloading to DATA_DIR.
    Configurable DATA_DIR and CACHE_DIR as env variables.
    """

    config = None

    def load(self, lang, model_class, force_download=False):
        """
        Loads tokenizer and model weights, with configuration files.
        Observes following resolution rules:
        - force_download==False: cache -> data_dir -> download from HuggingFace's repo.
        - force_download==True:  data_dir -> download from HuggingFace's repo.

        :param str lang: language code, eg., 'fr', 'en' to pick from `.config`.
            Must exist as a key in this `.config` class attribute!
        :param model_class: eg., `transformers.AutoModelForSeq2SeqLM`, etc.
        :param bool force_download: overrides the cached files if they exist,
            but resolution is remains: `data_dir` -> HF.

        Notes:
        ------
            (1) AutoTokenizer requires a config file, even if the tokenizer was found in the path??
                https://github.com/huggingface/transformers/issues/6368#issuecomment-671250169
                https://github.com/huggingface/transformers/issues/4197#issuecomment-625496433
        """
        model, tokenizer = None, None

        # assert required params
        assert self.config, "`ConfigLoader.config` required."
        assert lang in list(self.config), f"language not supported: {lang}"
        model_name = self.config[lang]["model"]
        tokenizer_name = self.config[lang]["tokenizer"]
        load_args = model_class, model_name, tokenizer_name

        if force_download:
            # here `force_download` kw to `from_pretrained()` is requesting to invalidate the cache.
            # files resolution: data_dir -> HuggingFace; and save downloaded to data_dir
            model, tokenizer = load_model(*load_args, save=True,
                                          force_download=force_download)

        else:
            # attempt finding the files in cache_dir on cache miss.
            # files resolution: cache -> data_dir ; but don't save (assumes files already inside data_dir)
            try:
                model, tokenizer = load_model(*load_args)

            # attempt downloading from HF (online=True) since the files are missing
            # from both cache AND data_dir (above)
            # TODO: should explicitly catch: ValueError, HFValidationError
            except Exception as e:
                mk_msg = lambda success: \
                    "couldn't find {files} in DATA_DIR={data_dir}, " \
                    "attempting download from HuggingFace ...{success}".format(
                        files=(tokenizer_name, model_name), data_dir=data_dir,
                        success=("OK" if success else "FAILED"))

                model, tokenizer = load_model(*load_args, online=True, save=True,
                                              force_download=force_download)
                print(mk_msg(bool(model and tokenizer)))

        # FIXME: to debug msg
        print('─' * 3 + f" {model_name} ")
        for envvar, path in dict(
            DATA_DIR=data_dir, CACHE_DIR=cache_dir or '~/.cache/huggingface/transformers/'
        ).items():
            r, errmsg = get_path_stats(path)
            log_msg = errmsg or \
                  "{envvar}\t ({size}, {file_count} files)\t -> {data_dir}"\
                      .format(envvar=envvar, data_dir=data_dir, **r)
            print(log_msg)

        return model, tokenizer


