from typing import Tuple

import torch
from transformers import AutoModelForSeq2SeqLM

from newsnlp.base import Pretrained
from newsnlp.logging import log

# truncate text to 1024 words, ie, barthez's max input seq len
# keep 70% by rule of thumbs, ie truncate abt 300/1024 words of
# 1024 long text to actually get to near the 1024 words limit
MAX_INPUT_LEN = 1024
MAX_INPUT_LEN_RATIO = .72

# max len of inferred summary.
# eg. 240 to accommodate a Tweet's len
SUM_TEXT_MAX_LEN = 1024
SUM_TITLE_MAX_LEN = 240


class TextSummarizer(Pretrained):
    """
    Multilanguage text summarizer.

    Supported languages:
    * French: uses pretrained [BARThez](https://github.com/moussaKam/BARThez)
        used on a summarization task (French text only, up to 1024 words)
    """
    max_length = SUM_TEXT_MAX_LEN
    config = {
        "fr": {
            "model": "moussaKam/barthez",
            "tokenizer": "moussaKam/barthez"
        }
    }

    def __init__(self, lang, max_length=None, **kwargs):
        """
        Initialize the summarizer.
        :param str lang: language code, eg. `fr`, `en` to pick from `.config`.
            Must exist as a key in this `.config` class attribute!
        :param int max_length:  *optional*, defaults to 20
            The maximum length of the sequence to be generated.
        TODO: build support for more languages
        """
        self.max_length = max_length or self.max_length
        assert lang == "fr", "French only supported as of now"

        self.model, self.tokenizer = \
            self.load(lang, AutoModelForSeq2SeqLM, **kwargs)

    def __call__(self, text):
        return self.summarize(text)

    def summarize(self, text, num_beams=4) -> Tuple[str, float]:
        """
        Text generation with beam search decoding of output sequence.
        This had proved more stable than greedy search, and certainly more accurable
        cf. transformers.generation.utils.GenerationMixin.compute_transition_scores.__doc__
        https://huggingface.co/docs/transformers/v4.32.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
        """

        summary, score = "", -1
        log_msg = f"summarizing text {len(text)} -> {self.max_length} chars"

        # inputs = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")
        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])

        # pretrained BARThez can encode sequences only up to 1024 words
        # cf., max_model_input_sizes= {'moussaKam2/mbarthez': 1024,
        #                              'moussaKam2/barthez': 1024,
        #                              'moussaKam2/barthez-orangesum-title': 1024}
        seq_len = input_ids.size(dim=1)
        if seq_len > MAX_INPUT_LEN:
            input_ids = input_ids[:, :MAX_INPUT_LEN]
            log.debug(f"input sequence length {seq_len} greater than model capacity, "
                      f"truncating to {MAX_INPUT_LEN}")

        try:
            outputs = self.model.generate(
                input_ids, return_dict_in_generate=True, output_scores=True,
                renormalize_logits=True,  max_new_tokens=self.max_length-1, num_beams=num_beams)
            summary = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        except Exception:
            log.exception(log_msg + ": summary generation failed")

        else:
            try:
                # extract log probs for selected sequence (first sequence)
                # FIXME: forbids `max_length` arg in .generate()
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
                score = transition_scores[0].exp().prod().item()
            except Exception as e:
                log.exception(log_msg + ": score computation failed")

        return summary, score


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
