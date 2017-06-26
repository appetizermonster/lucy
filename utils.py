import numpy as np
from numpy.linalg import norm


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def sent2labels(intent, sentence):
    return [intent.predict_entity_label(x) for x in sentence.words]


def sent2features(intent, sentence):
    return [word2features(intent, sentence, i) for i in range(len(sentence.words))]


def has_number(str):
    return any(x.isdigit() for x in str)


def word2features(intent, sentence, i):
    words = sentence.words
    postags = sentence.postags
    word = words[i]
    label = intent.predict_entity_label(word)
    features = {
        'bias': 1.0,
        'label': label,
        'label[:2]': label[:2],
        'label[:3]': label[:3],
        'has_number': has_number(word)
    }
    if i > 0:
        word_p = words[i - 1]
        label_p = intent.predict_entity_label(word_p)
        postag_p = postags[i - 1]
        features.update({
            '-1:word': word_p,
            '-1:label': label_p,
            '-1:postag': postag_p,
            '-1:has_number': has_number(word_p)
        })
    else:
        features['BOS'] = True

    if i < len(words) - 1:
        word_n = words[i + 1]
        label_n = intent.predict_entity_label(word_n)
        postag_n = postags[i + 1]
        features.update({
            '+1:word': word_n,
            '+1:label': label_n,
            '+1:postag': postag_n,
            '+1:has_number': has_number(word_n)
        })
    else:
        features['EOS'] = True

    return features
