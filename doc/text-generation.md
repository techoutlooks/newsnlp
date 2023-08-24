# Text Generation


Text Generation with HF's Transformers
Summarization models subclass transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration


[Lecture](https://huggingface.co/blog/how-to-generate)
[API Ref](https://huggingface.co/docs/transformers/main_classes/text_generation)
[Transformers from Scratch](https://zhuanlan.zhihu.com/p/625798971)
[Methods for generating text with GPT-2](https://christianjmills.com/posts/transformers-book-notes/chapter-5/index.html)


### Nota

* BERT is a masked language model, different from classical (autoregressive or causal language models) language models.
Perplexity (PPL), one of the most common metrics for evaluating language models, not well-defined for masked models.
cf. [Models summary](https://huggingface.co/docs/transformers/model_summary), 
[perplexity](https://huggingface.co/docs/transformers/perplexity)

### model.generate()

https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14


* `return_dict_in_generate=True` returns ['sequences'], but together with `output_scores=True`, 
  it returns ['sequences', 'sequences_scores', 'scores']. 

* `.sequences` contains the BOS token, which is not predicted by the model. 
  [src](https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/10)

* `.scores` gives the logits for all batches/beams at each step.

  - Scores are the UNNORMALIZED log probabilities (they will only be normalized if you pass `renormalize_logits=True` to .generate()). 
  It is a tuple containing one entry for each generated token. Each tuple member is a tensor containing the log probabilities 
  from the model, for all words in the vocabulary. 
  These log probabilities are gathered AFTER manipulating them with our logits processors 
  (e.g. after setting the probability of certain words to 0, after applying top k, …);

  - when using the beam search to generate text, each of the elements in the tuple `.scores` contains a matrix, 
  where each row corresponds to each beam, stored at this step, while the values are the sum of log-probas of 
  the previous sequence and the next token [src](https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175/18)

* `sequence_scores` correspond to the sum of the log probabilities in scores. 
If length_penalty is not 0, sequence_scores will be divided by `sequence_length**length_penalty`


### model.compute_transition_scores()

* Doc
  - [ref](https://huggingface.co/docs/transformers/main_classes/text_generation)
  - https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/34


* only needed to get the logits of the selected tokens, the tokens present in outputs.sequences.
	https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/21

* `transition_scores` contains scores (log probabilities) for the tokens that were selected at generation time. 
You can set `normalize_logits=True` to ensure they are normalized at a token level (i.e. to ensure 
the sum of probabilities for all vocabulary at a given generation step is 1).
If interested in probabilities, normalize_logits should be True, so that the probability of all possible generated sequences sum to 1

* [Example](https://huggingface.co/docs/transformers/main_classes/text_generation)
```python
import numpy as np

# this yields output scores of 4 sequences by default (num_return_sequences=4)
transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

# three ways of computing the generated sequence's  probability
score = np.exp(transition_scores[0].numpy()).prod()
score = transition_scores.sum(axis=1).exp().sum().item()
score = transition_scores[0].exp().prod().item()
```


    # same as below --
    predict2 = self.model.generate(input_ids, max_length=self.max_length,  output_scores=True, return_dict_in_generate=True)
    # assert predict2['sequences'] == predict2[0] = predict[0]

    #  sequences_scores: Final beam scores of the generated sequences.
    # I would have expected the 'sequences_scores' to be equal to the product of the probabilities
    # of each generated token conditional on the previous generated tokens:
    # assert predict2['sequences_scores'] == predict2[1] # == tensor([-0.2984])


    # `labels=input_ids` to include the loss in model output
    model(input_ids, labels=input_ids).loss.item()


