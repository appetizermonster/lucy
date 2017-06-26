import re
from collections import namedtuple

from sentence import *

TAG_RE = r'<([a-zA-Z0-9_]+)>(.*?)</\1>'

ExampleEntity = namedtuple('ExampleEntity', ['name', 'start', 'length'])


class Example:
    def __init__(self, example_text):
        self._example_text = example_text
        self._parse()

    def _parse(self):
        labels = _get_labels(self._example_text)
        self.text = _strip_tags(self._example_text)

        self.sentence = Sentence(self.text)
        self.labels = [self._find_label(labels, x.offset)
                       for x in self.sentence.tokens]

    def _find_label(self, entities, start):
        for entity in entities:
            e_start = entity.start
            e_end = e_start + entity.length - 1
            if e_start <= start <= e_end:
                return entity.name
        return '_'


def _strip_tags(text):
    return re.sub(TAG_RE, r'\2', text)


def _get_labels(text):
    entities = []
    match_offset = 0
    matches = re.finditer(TAG_RE, text)
    for match in matches:
        label = match.group(1)
        tag_size = (len(label) * 2) + 5

        start = match.start() - match_offset
        length = match.end() - match.start() - tag_size

        entities.append(ExampleEntity(label, start, length))
        match_offset += tag_size
    return entities
