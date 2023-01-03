from newsnlp import TfidfVectorizer


def run():

    # LSG
    lsg_v1 = "Lève-toi, sois éclairée, car ta lumière arrive, Et la gloire de l'Eternel se lève sur toi."
    lsg_v2 = "Voici, les ténèbres couvrent la terre, Et l'obscurité les peuples; Mais sur toi l'Eternel se lève, Sur toi sa gloire apparaît."
    lsg_v3 = "Des nations marchent à ta lumière, Et des rois à la clarté de tes rayons."

    # Martin
    mar_v1 = "Lève-toi, sois illuminée; car ta lumière est venue, et la gloire de l'Eternel s'est levée sur toi."
    mar_v2 = "Car voici, les ténèbres couvriront la terre, et l'obscurité couvrira les peuples; mais l'Eternel se lèvera sur toi, et sa gloire paraîtra sur toi."
    mar_v3 = "Et les nations marcheront à ta lumière, et les Rois à la splendeur qui se lèvera sur toi."

    # Darby
    dar_v1 = "Leve-toi, resplendis, car ta lumiere est venue, et la gloire de l'Eternel s'est levee sur toi."
    dar_v2 = "Car voici, les tenebres couvriront la terre, et l'obscurite profonde, les peuples; mais sur toi se levera l'Eternel, et sa gloire sera vue sur toi."
    dar_v3 = "Et les nations marcheront à ta lumiere, et les rois, à la splendeur de ton lever."

    # my TfidfVectorizer
    # =================
    # single text/doc, produces 1D (vocab_size,) doc row vector
    # text = [lsg_v1 + lsg_v2 + lsg_v3]
    # tfidf = TfidfVectorizer()
    # doc_vec = tfidf(text)

    # in: 3 corpora,  3 doc set each
    # out a (9,vocab_size) shaped dataset
    lsg, martin, darby = (lsg_v1, lsg_v2, lsg_v3), (mar_v1, mar_v2, mar_v3), (dar_v1, dar_v2, dar_v3)
    data = lsg + martin + darby
    tfidf = TfidfVectorizer()
    Q = tfidf(data).fit_transform()
    cosine_sim = tfidf(data).fit().cosine_similarity()
    most_similar = tfidf(data).fit().similar_to(1, 0.3)
    pass

    # sklearn's TfidfVectorizer
    # ========================
    from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVectorizer2
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf2 = TfidfVectorizer2()
    docs = [' '.join([t for t in d]) for d in tfidf.corpus]
    Q2 = tfidf2.fit_transform(docs)
    cosine_sim2 = cosine_similarity(Q2, Q2)
    pass


if __name__ == '__main__':
    run()

