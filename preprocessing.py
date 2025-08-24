from razdel import tokenize
from string import punctuation
from nltk.stem.snowball import SnowballStemmer

snowball = SnowballStemmer(language='russian')
puncts = set(punctuation + '«»')

def tokenize_sentence(sentence):
    return [snowball.stem(t.text) for t in tokenize(sentence) if t.text not in puncts]
