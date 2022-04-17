import spacy


import nltk
from nltk.corpus import stopwords
from clutrr.preprocess import Instance

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")


def tokenize_normalize_naive(instance: Instance, vocab: list[str]) -> tuple[list[str], set[str]]:
    doc = nlp(instance.raw_story)
    normalized_tokens = []
    entities = []
    reading_entity = False

    sanity_check = 0

    busted = []
    for token in doc:
        if token.text in "[]":
            # toggle
            reading_entity = not reading_entity
            sanity_check += 1
            continue
        if token.is_alpha:
            normalized = token.lower_.strip()
            if normalized in vocab or reading_entity:
                # append only tokens for which embeddings exist
                normalized_tokens.append(normalized)
            if normalized in stopwords:
                busted.append(normalized)
            if reading_entity:
                entities.append(normalized)

    if not sanity_check == 2 * len(entities):
        assert False

    return normalized_tokens, set(entities)


def tokenize_normalize_wo_stopwords(instance: Instance, vocab: list[str]) -> tuple[list[str], set[str]]:
    doc = nlp(instance.raw_story)
    normalized_tokens = []
    entities = []
    reading_entity = False

    sanity_check = 0

    busted = []
    for token in doc:
        if token.text in "[]":
            # toggle
            reading_entity = not reading_entity
            sanity_check += 1
            continue
        if token.is_alpha:
            normalized = token.lower_.strip()
            if (
                normalized in vocab or reading_entity
            ) and normalized not in stopwords:
                # append only tokens for which embeddings exist
                normalized_tokens.append(normalized)
            if normalized in stopwords:
                busted.append(normalized)
            if reading_entity:
                entities.append(normalized)

    if not sanity_check == 2 * len(entities):
        assert False

    return normalized_tokens, set(entities)


def tokenize_normalize_relevant(instance: Instance, vocab: list[str]) -> tuple[list[str], set[str]]:
    doc = nlp(instance.raw_story)
    normalized_tokens = []
    entities = []
    reading_entity = False

    sanity_check = 0

    busted = []
    for token in doc:
        if token.text in "[]":
            # toggle
            reading_entity = not reading_entity
            sanity_check += 1
            continue
        if token.is_alpha:
            normalized = token.lower_.strip()
            if normalized in vocab or reading_entity:
                # append only tokens for which embeddings exist
                normalized_tokens.append(normalized)
            if normalized in stopwords:
                busted.append(normalized)
            if reading_entity:
                entities.append(normalized)

    if not sanity_check == 2 * len(entities):
        assert False

    return normalized_tokens, set(entities)
