
# https://huggingface.co/moussaKam/barthez-orangesum-abstract
# http://master2-bigdata.polytechnique.fr/barthez
import json

from newsnlp.categorizer import Categorizer
from newsnlp.summarizer import TextSummarizer, TitleSummarizer


# truncate text to 1024 words, ie, barthez's max input seq len
# keep 70% by rule of thumbs, ie truncate abt 300/1024 words of
# 1024 long text to actually get to near the 1024 words limit
MAX_INPUT_LEN_RATIO = .7

# max len of inferred summary.
# eg. 240 to accommodate a Tweet's len
SUM_MAX_LEN = 240

# ony French supported as of May 8th 2022
LANGUAGE = "fr"


def load_data(path):
    with open(path) as f:
        data = json.load(f)
        texts = [item['text'] for item in data]
        return texts


if __name__ == '__main__':

    data = load_data("data/data.json")

    text = [w for text in data for w in text.split(' ')]
    seq_len = len(text)
    text = text[:round(seq_len * MAX_INPUT_LEN_RATIO)]
    text = " ".join(text)

    summary = TextSummarizer(lang=LANGUAGE)(text)
    title = TitleSummarizer(lang=LANGUAGE)(text)

    topics = Categorizer(lang=LANGUAGE)(text)
    category = topics[0][0] if topics else 'N/A'

    print(f"\ncategory: {category}")
    print(f"\ntitle: {title}")
    print(f"summary ({len(summary)}) words: {summary}")


# yields:
# category: Societe
# title: Guineematin.com : les magistrats nommés à la Cour suprême
# summary (7) words: Le chef de l’Etat, le colonel Mamadi Doumbouya, a fait l’acquisition de plusieurs postes dans l’administration judiciaire guinéenne.

