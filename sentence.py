from lib import koreantext

koreantext.init()
tagger = koreantext.Tagger()


class Sentence:
    def __init__(self, text):
        self.tokens = tagger.tokenize(text)
        self.postags = [x.pos for x in self.tokens]
        self.words = [x.text for x in self.tokens]
