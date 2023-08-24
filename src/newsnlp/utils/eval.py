from pprint import pprint
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# TODO: Scoring using perplexity
# https://huggingface.co/docs/transformers/perplexity
# https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/10
# https://gist.github.com/yuchenlin/eb63e2d0513f70cfc9bb85fa5a78953b
# https://github.com/huggingface/transformers/issues/473
# https://stackoverflow.com/a/65242898


def to_logprobs(model, input_ids):
    """
    Get probability for generated sequence manually.
    * Beware of inconsistency between this implementation and the official way of evaluating probabilities,
     i.e., model.generate() + model.compute_transition_scores().
    [24801](https://github.com/huggingface/transformers/issues/24801#issue-1802087328)
    """

    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # these models return the probabilities for the next token, so
    # pairing between the probabilities and the tokens has to be shifted by one
    # probability at index 0 corresponds to the token at index 1 (token after BOS)
    input_ids = input_ids[:, 1:]    # EC: drops the BOS token
    probs = probs[:, :-1, :]        # EC: removes last prediction to keep lengths the same

    # EC: re-orders the probabilities according to tokens order
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    return gen_probs


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    """
    Get probabilities for generated output tokens [by joaogante@HF](
    https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15)
    This is similar to [Beam Search Decoding For Text Generation In Python](
    https://medium.com/geekculture/beam-search-decoding-for-text-generation-in-python-9184699f0120)
    """
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    gen_probs = to_logprobs(model, input_ids)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)

    return batch
