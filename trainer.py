import pickle

import sklearn_crfsuite
from sklearn.ensemble import *

import utils
import fileutils
from vectorizer import *


class Trainer:
    def __init__(self, wordvec, filename):
        self._wordvec = wordvec
        self._filename = filename

    def train(self, intents):
        fileutils.ensuredir(self._filename)

        self._train_intents(intents)
        self._train_entities(intents)

    def _train_intents(self, intents):
        vectorizer = WordVecEmbeddingVectorizer(self._wordvec)
        classifier = ExtraTreesClassifier(n_estimators=200, random_state=1)
        intents_words = []
        intent_names = []
        for intent in intents:
            for example in intent.examples:
                intents_words.append(example.sentence.words)
                intent_names.append(intent.name)
        text_vecs = vectorizer.fit_transform(intents_words)
        classifier.fit(text_vecs, intent_names)

        # dump classifier
        with open('%s.intent.clf' % self._filename, 'wb') as f:
            pickle.dump(classifier, f)

    def _train_entities(self, intents):
        for intent in intents:
            crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                       max_iterations=100,
                                       all_possible_transitions=True
                                       )
            filename = '%s.%s.entity' % (self._filename, intent.name)
            features_list = []
            labels_list = []
            for example in intent.examples:
                sentence = example.sentence
                sent_features = utils.sent2features(intent, sentence)
                # predicted_labels = utils.sent2labels(intent, sentence)
                sent_labels = example.labels
                features_list.append(sent_features)
                labels_list.append(sent_labels)
            crf.fit(features_list, labels_list)

            with open(filename, 'wb') as f:
                pickle.dump(crf, f)
