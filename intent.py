import numpy as np

from example import Example
from utils import cosine_similarity


class Intent:
    def __init__(self, wordvec, name, raw_examples, entities):
        self.wordvec = wordvec
        self.name = name
        self._raw_examples = raw_examples
        self.entities = entities
        self._entity_vectors = {}

        self._make_examples()
        self._fill_nonamed_entities()
        self._init_entity_vectors()

    def _make_examples(self):
        examples = []
        for raw_example in self._raw_examples:
            example = Example(raw_example)
            examples.append(example)
        self.examples = examples

    def _fill_nonamed_entities(self):
        nonamed_entity_values = self.entities.get('_', [])
        for example in self.examples:
            words = example.sentence.words
            labels = example.labels
            for i in range(len(words)):
                label = labels[i]
                if label is not '_':
                    continue
                word = words[i]
                nonamed_entity_values.append(word)
        self.entities['_'] = list(set(nonamed_entity_values))

    def _init_entity_vectors(self):
        for label, values in self.entities.items():
            mean_vec = np.mean(
                [self.wordvec[x] for x in values], axis=0)
            self._entity_vectors[label] = mean_vec

    def predict_entity_label(self, word):
        word_vec = self.wordvec[word]
        selected = (0, None)
        for label, vector in self._entity_vectors.items():
            cos_sim = cosine_similarity(word_vec, vector)
            if cos_sim > selected[0]:
                selected = (cos_sim, label)
        return selected[1] or '_'


def json2intents(wordvec, obj):
    intents = []
    for key, value in obj.items():
        raw_examples = value['examples']
        entities = value.get('entities', {})
        intents.append(Intent(wordvec, key, raw_examples, entities))
    return intents
