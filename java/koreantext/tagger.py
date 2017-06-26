from jpype import *
from .utils import *
from collections import namedtuple

KoreanToken = namedtuple('KoreanToken', ['text', 'pos', 'offset', 'length'])


class Tagger:
    def __init__(self):
        self._java_cls = JClass(
            'org.openkoreantext.processor.OpenKoreanTextProcessorJava')

    def normalize(self, text):
        return self._java_cls.normalize(to_jstring(text))

    def tokenize(self, text):
        s_tokens = self._java_cls.tokenize(to_jstring(text))
        j_tokens = self._java_cls.tokensToJavaKoreanTokenList(s_tokens)
        tokens = [
            KoreanToken(x.getText(), x.getPos().name(),
                        x.getOffset(), x.getLength())
            for x in j_tokens
        ]
        return tokens
