
## Features

* Support for conda

## Dev

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

### Trying pretrained models from Huggingface 

- Download once pretrained models locally, suitable for the summarisation task
```shell
from transformers import AutoTokenizer

# load pretrained models from checkpoints
tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract")

# save locally
tokenizer.save_pretrained('data/barthez-barthez-tokenizer')
model.save_pretrained('data/barthez')

```

### Using this library

- Env setup



```shell
# pip-sync

conda create --name newsbot python=3

```

- Usage
```shell
# set the downloads folder for pretrained models (optional). 
# defaults to `newsnlp/data` in the source tree.
#NLP_DATA_DIR=/mnt/gcs
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