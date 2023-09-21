
Python library that transforms news data using a variety of NLP-based processing tasks. 
You may:

* Generate summary, caption, category for scraped content using NLP.
* Gather posts by affinity, eg. all similar titles from a bunch of websites.
* Generate meta-summaries from posts grouped by affinty (concept of sibling posts) using NLP.
* Remove ads and undesirable content using NLP.
* Downloads and caches pre-trained Transformer models from HuggingFace for the summarization tasks (check supported languages),
* Supports Anaconda envs for scientific computing

## Features 

- Uses [Playwright](https://playwright.dev/) to inflat dynamic content from websites (eg. Ads, JavaScript) before processing.
- Downloads and caches pretrained NLP models locally, suitable for fast inference.
- Pretrained deep learning Ad detection model.
- Pretrained Transformer-based LLM for the summarization tasks (only French supported yet).
- Ad detection implements the (Kushmerick, 1999)](./doc/kushmerick99learning.pdf) paper partially,
  but relies on Deep Learning rather than statistical fitting.

## Dev

- Env setup

```shell
conda create --name newsbot python=3
pip-sync
```

###
- Summariser's dependencies: `sentencepiece`
```shell

conda install -c conda-forge sentencepiece
conda install pytorch torchvision -c pytorch
#conda install -c conda-forge transformers
conda install -c huggingface transformers
```
- TFIDF deps: `` 
```shell

conda install -c conda-forge spacy
python -m spacy download en_core_web_sm

```


## TODO

### Optimisations

- Use optimised TF-IDF from Spacy or SkLearn
- Utilise only half of the symmetric TF-IDF matrix
- Resume vectorization of corpus where last task left off.
  This implies saving vectorization result to disk, and merging with docs newly added to the db. 
- Cython ??

## Feature request

- Multiple languages support - OK O7/06/2023
- Summarizer currently only supports 1024 words max. Find more powerful model? push model capacity?
- Require `conda` in setuptools/pyproject.toml?
- WTF is the [sumy](https://github.com/miso-belica/sumy) summariser?
