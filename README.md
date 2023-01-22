

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
pip-sync
```

- Usage
```shell
# set the downloads folder for pretrained models (optional). 
# defaults to `newsnlp/data` in the source tree.
#DATA_DIR=/mnt/gcs
```

## TODO

- Dockerfile with `conda`, means also upgrade projects consuming the lib to conda
- WTF is the [sumy](https://github.com/miso-belica/sumy) summariser?