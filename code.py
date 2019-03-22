#from __future__ import unicode_literals, print_function, division
import numpy
import scipy
import matplotlib
import pandas
import statsmodels
import sklearn
import theano
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
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
        pairs[i].append(english_lines[i])
        pairs[i].append(hindi_lines[i])
    return pairs

def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

english_text = load_doc('english.txt')
hindi_text = load_doc('hindi.txt')
pairs = to_pairs(english_text, hindi_text)
clean_pairs = clean_pairs(pairs)

en_samples = []
de_samples = []

# Define impty sets to store the characters in them:
en_chars = set()
de_chars = set()

# Split the samples and get the character sets :
for line in text:
    en_ , de_ = line.split('\t')
    de_ = '\t' + de_
    print (en_)
    print (de_)
    for char in de_:
        if char not in de_chars:
            de_chars.add(char)
    for char in en_:
        if char not in en_chars:
            en_chars.add(char)
    en_samples.append(en_)
    de_samples.append(de_)

de_chars.add('\n')
de_chars.add('\t')

en_chars.add('\n')
en_chars.add('\t') 
