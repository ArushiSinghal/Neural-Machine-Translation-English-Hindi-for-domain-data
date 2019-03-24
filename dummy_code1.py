#from __future__ import unicode_literals, print_function, division
import numpy
import scipy
import matplotlib
import pandas
import statsmodels
import sklearn
import tensorflow
import keras
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
INDIC_NLP_LIB_HOME=r"/home/arushi/Neural-Machine-Translation/anoopkunchukuttan-indic_nlp_library-eccde81/src"
INDIC_NLP_RESOURCES=r"/home/arushi/Neural-Machine-Translation/indic_nlp_resources-master"
import sys
sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from pickle import dump
from unicodedata import normalize
from numpy import array


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        #index of each new word
        self.word2index = {}
        #count of that word
        self.word2count = {}
        #index with word
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

def to_pairs(english_text, hindi_text):
    english_lines = english_text.strip().split('\n')
    hindi_lines = hindi_text.strip().split('\n')
    pairs = []
    for i in range(len(hindi_lines)):
        pairs.append([])
        pairs[i].append(pre_process_english_sentence(english_lines[i]))
        pairs[i].append(pre_process_hindi_sentence(hindi_lines[i]))
    return pairs

def clean_text(line):
    text = line
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",'')
    text=text.replace(u"-",'')
    text=text.replace(u"।",'')
    text=text.replace(u"?",'')
    text=text.replace(u"\\",'')
    text=text.replace(u"_",'')
    text=re.sub('[a-zA-Z]', '', text)
    return text

def pre_process_english_sentence(line):
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    line = line.split()
    line = [word.lower() for word in line]
    line = [word.translate(table) for word in line]
    line = [re_print.sub('', w) for w in line]
    line = [word for word in line if word.isalpha()]
    line = ' '.join(line)
    return line

def pre_process_hindi_sentence(line):
    remove_nuktas=False
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer("hi",remove_nuktas)
    line=normalizer.normalize(line)
    line=clean_text(line)
    tokens = list()
    for t in indic_tokenize.trivial_tokenize(line):
        tokens.append(t)
    line = tokens
    line = [word.lower() for word in line]
    line = [word for word in line if not re.search(r'\d', word)]
    line = ' '.join(line)
    return (line)

def prepareData(pairs):
    input_lang = Lang('eng')
    output_lang = Lang('hin')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang

def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

english_text = load_doc('English')
hindi_text = load_doc('Hindi')
pairs = to_pairs(english_text, hindi_text)
input_lang, output_lang = prepareData(pairs)

print (len(pairs))
save_clean_data(array(pairs), 'english-hindi-dummy.pkl')
raw_dataset = load_clean_sentences('english-hindi-dummy.pkl')
n_sentences = len(pairs)
print (type(n_sentences))
dataset = raw_dataset[:n_sentences, :]
#divide dataset into train and test
n_sentences = int((len(pairs))/2)
dataset_train = dataset[:n_sentences]
dataset_test = dataset[n_sentences:]
# random shuffle
shuffle(dataset_train)
shuffle(dataset_test)
# save
save_clean_data(dataset, 'english-hindi-both.pkl')
save_clean_data(dataset_train, 'english-hindi-train.pkl')
save_clean_data(dataset_test, 'english-hindi-test.pkl')

dataset = load_clean_sentences('english-hindi-both.pkl')
train = load_clean_sentences('english-hindi-train.pkl')
test = load_clean_sentences('english-hindi-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
print (eng_tokenizer)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
print (eng_tokenizer.word_index)
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare hindi tokenizer
hin_tokenizer = create_tokenizer(dataset[:, 1])
hin_vocab_size = len(hin_tokenizer.word_index) + 1
print (hin_tokenizer.word_index)
hin_length = max_length(dataset[:, 1])
print('Hindi Vocabulary Size: %d' % hin_vocab_size)
print('HIndi Max Length: %d' % (hin_length))
