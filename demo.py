import json
import pathlib

import fasttext

from intent import *
from parser import Parser
from sentence import Sentence
from trainer import Trainer
import utils

wordvec = fasttext.load_model('fasttext/ko.bin')
model_json = pathlib.Path('model.json').read_text()
intents = json2intents(wordvec, json.loads(model_json))

MODEL_PATH = 'trained/test'

# training
trainer = Trainer(wordvec, MODEL_PATH)
trainer.train(intents)

# parsing
sentence = Sentence('내일 서울 날씨 알려줘')
parser = Parser(MODEL_PATH, wordvec, intents)
result = parser.parse(sentence)
print(result)
