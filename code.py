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
    text = file.read()
    file.close()
    return text

def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs
