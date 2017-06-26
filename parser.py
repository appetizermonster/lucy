import pickle
from collections import namedtuple

import utils
from vectorizer import *

ParseResult = namedtuple('ParseResult', ['intent_name', 'entities'])
LabeledEntity = namedtuple('LabeledEntity', ['text', 'words'])


class Parser:
    def __init__(self, filename, wordvec, intents, threshold=0.3):
        with open('%s.intent.clf' % filename, 'rb') as f:
            self._intent_clf = pickle.load(f)

        self._wordvec = wordvec
        self._filename = filename
        self._intents = intents
        self.threshold = threshold

    def parse(self, sentence):
        intent_name = self._predict_intent(sentence)
        if intent_name is None:
            return None
        intent = next(x for x in self._intents if x.name == intent_name)
        entity_labels = self._extract_entitiy_labels(intent, sentence)
        entities = self._make_entities(sentence.tokens, entity_labels)
        return ParseResult(intent_name, entities)

    def _predict_intent(self, sentence):
        vectorizer = WordVecEmbeddingVectorizer(self._wordvec)
        T = vectorizer.transform([sentence.words])
        probas = list(self._intent_clf.predict_proba(T)[0])
        max_proba = max(probas)
        if max_proba < self.threshold:
            return None
        max_idx = probas.index(max_proba)
        return self._intent_clf.classes_[max_idx]

    def _extract_entitiy_labels(self, intent, sentence):
        with open('%s.%s.entity' % (self._filename, intent.name), 'rb') as f:
            crf = pickle.load(f)

        features = utils.sent2features(intent, sentence)
        labels = crf.predict([features])[0]
        return labels

    def _make_entities(self, tokens, entity_labels):
        entity_last_poses = {}
        entity_texts = {}
        entity_words = {}
        for token, label in zip(tokens, entity_labels):
            if label is '_':
                continue
            entity_last_pos = entity_last_poses.get(label, -1)
            need_space = (entity_last_pos >= 0) and (
                token.offset > entity_last_pos)
            entity_text = entity_texts.get(label, '')
            entity_word = entity_words.get(label, [])
            if need_space:
                entity_text += ' '
            entity_text += token.text
            entity_word.append(token.text)
            entity_last_poses[label] = token.offset + token.length
            entity_texts[label] = entity_text
            entity_words[label] = entity_word
        entities = {
            x: LabeledEntity(entity_texts[x], entity_words[x]) for x in entity_texts.keys()
        }
        return entities
