import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp.vocab['wolf'].vector
word2 = nlp.vocab['dog'].vector
word3 = nlp.vocab['cat'].vector

from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)


def vector_math(word1, word2, word3):
    """
    Performs vector arithmetic on given words.

    Parameters
    ----------
    word1 : str
    word2 : str
    word3 : str

    Returns
    -------
    list of tuples
        Top 10 closest words result of the vector arithmetic, sorted by cosine distance.

    """
    result_vector = nlp.vocab[word1].vector - nlp.vocab[word2].vector + nlp.vocab[word3].vector
    similarity_score = []

    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity_score.append((word.text, cosine_similarity(word.vector, result_vector)))

    similarity_score = sorted(similarity_score, key=lambda item: -item[1])

    return similarity_score[:10]


vector_math('king','man','woman')