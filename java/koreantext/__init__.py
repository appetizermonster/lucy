from . import jvm
from . import tagger


def init():
    jvm.init_jvm()


Tagger = tagger.Tagger
