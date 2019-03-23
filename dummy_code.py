import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
INDIC_NLP_LIB_HOME=r"/home/arushi/Neural-Machine-Translation/anoopkunchukuttan-indic_nlp_library-eccde81/src"
INDIC_NLP_RESOURCES=r"/home/arushi/Neural-Machine-Translation/indic_nlp_resources-master"
import sys
sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

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

def pre_process_hindi_sentence(line):
    print (line)
    remove_nuktas=False
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer("hi",remove_nuktas)
    output_text=normalizer.normalize(line)
    print (output_text)
    #line = line.split()
    #print (line)
    #line = ' '.join(line)
    #print (line)
    return output_text

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

english_text = load_doc('English')
hindi_text = load_doc('Hindi')
#hindi_text = load_doc('French.txt')
pairs = to_pairs(english_text, hindi_text)
#print (pairs)
