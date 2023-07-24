import en_core_web_sm
import fr_core_news_sm


# cache nlp pipeline
# takes time to load !
_nlp = None


def load_nlp(lang, use_cache=True):
    global _nlp
    if not _nlp or use_cache:
        _nlp = {
            'en': en_core_web_sm,
            'fr': fr_core_news_sm
        }[lang or 'en'].load()
    return _nlp