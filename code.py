import scipy
import numpy
import matplotlib
import pandas
import statsmodels
import sklearn
import theano
import tensorflow
import keras

def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.readlines()
    file.close()
    return text

text_lines = load_doc('a.txt')
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
