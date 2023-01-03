from os.path import dirname, realpath

import torch
from transformers import AutoModelForSeq2SeqLM


# truncate text to 1024 words, ie, barthez's max input seq len
# keep 70% by rule of thumbs, ie truncate abt 300/1024 words of
# 1024 long text to actually get to near the 1024 words limit
from newsnlp.base import ConfigLoader

MAX_INPUT_LEN = 1024
MAX_INPUT_LEN_RATIO = .72

# max len of inferred summary.
# eg. 240 to accommodate a Tweet's len
SUM_TEXT_MAX_LEN = 1024
SUM_TITLE_MAX_LEN = 240


class TextSummarizer(ConfigLoader):
    """
    Pretrained [BARThez](https://github.com/moussaKam/BARThez)
    used on a summarization task (French text only, up to 1024 words)
    """
    max_length = SUM_TEXT_MAX_LEN
    config = {
        "fr": {
            "model": "moussaKam/barthez",
            "tokenizer": "moussaKam/barthez-tokenizer"
        }
    }

    def __init__(self, lang, max_length=None):
        """
            Initialize the summarizer.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
        """
        self.max_length = max_length or self.max_length
        assert lang == "fr", "French only supported as of now"

        self.model, self.tokenizer = \
            self.load_from_config(lang, AutoModelForSeq2SeqLM)

    def __call__(self, text):
        return self.summarize(text)

    def summarize(self, text):

        input_ids = torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=True)]
        )

        # pretrained BARThez can encode sequences only up to 1024 words
        # cf., max_model_input_sizes= {'moussaKam2/mbarthez': 1024,
        #                              'moussaKam2/barthez': 1024,
        #                              'moussaKam2/barthez-orangesum-title': 1024}
        seq_len = input_ids.size(dim=1)
        if seq_len > MAX_INPUT_LEN:
            input_ids = input_ids[:, :MAX_INPUT_LEN]
            print(f"input sequence length {seq_len} greater than model capacity. " 
                  f"truncating to {MAX_INPUT_LEN}")

        predict = self.model.generate(input_ids, max_length=self.max_length)[0]
        summary = self.tokenizer.decode(predict, skip_special_tokens=True)

        return summary


class TitleSummarizer(TextSummarizer):
    """
    https://huggingface.co/moussaKam/barthez-orangesum-title
    """

    max_length = SUM_TITLE_MAX_LEN
    config = {
        "fr": {
            "model": "moussaKam/barthez-orangesum-title",
            "tokenizer": "moussaKam/barthez-orangesum-title"
        }
    }
