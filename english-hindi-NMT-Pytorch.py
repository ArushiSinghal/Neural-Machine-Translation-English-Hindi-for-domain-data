#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from __future__ import unicode_literals, print_function, division
from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense, CuDNNLSTM, TimeDistributed, Flatten, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
from numpy import array, argmax
from numpy.random import rand, shuffle
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from keras.utils import multi_gpu_model
import scipy
import statsmodels
import sklearn
import tensorflow
import keras
from io import open
import unicodedata
import random
import math
import time
import os
import time
import re
import sys
from unicodedata import normalize
from numpy import array
from pickle import dump, load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.layers import RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INDIC_NLP_LIB_HOME=r"/home/arushi/Neural-Machine-Translation/anoopkunchukuttan-indic_nlp_library-eccde81/src"
INDIC_NLP_RESOURCES=r"/home/arushi/Neural-Machine-Translation/indic_nlp_resources-master"
sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

MAX_LENGTH = 100


# In[ ]:


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


# In[ ]:


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

def clean_text(text):
    text = text.replace(u',','')
    text = text.replace(u'"','')
    text = text.replace(u'"','')
    text = text.replace(u"‘‘",'')
    text = text.replace(u"’’",'')
    text = text.replace(u"''",'')
    text = text.replace(u"।",'')
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
    text=text.replace("'", "")
    text=text.replace('"', "")
    text= re.sub("'", '', text)
    text= re.sub("’", '', text)
    text=re.sub('[0-9+\-*/.%]', '', text)
    text=text.strip()
    text=re.sub(' +', ' ',text)
    exclude = set(string.punctuation)
    text= ''.join(ch for ch in text if ch not in exclude)
    return text

def pre_process_english_sentence(line):
    line = line.lower()
    line = clean_text(line)
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    line = line.split()
    line = [re_print.sub('', w) for w in line]
    line = [word for word in line if word.isalpha()]
    line = ' '.join(line)
    return line

def pre_process_hindi_sentence(line):
    line=re.sub('[a-zA-Z]', '', line)
    line = clean_text(line)
    remove_nuktas = False
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("hi",remove_nuktas)
    line = normalizer.normalize(line)
    tokens = list()
    for t in indic_tokenize.trivial_tokenize(line):
        tokens.append(t)
    line = tokens
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

english_text = load_doc('English')
hindi_text = load_doc('Hindi')
pairs_all = to_pairs(english_text, hindi_text)
input_lang, output_lang = prepareData(pairs_all)

df = pd.DataFrame(pairs_all)
df.columns = ["eng", "mar"]
lines = df
lenght_list=[]
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))
max_length_src = np.max(lenght_list)
print (max_length_src)
lenght_list=[]
for l in lines.mar:
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)
print (max_length_tar)

print(random.choice(pairs_all))
print(random.choice(pairs_all))
print(random.choice(pairs_all))
print(random.choice(pairs_all))


english_text = load_doc('testing_english.txt')
hindi_text = load_doc('testing_hindi.txt')
pairs_testing = to_pairs(english_text, hindi_text)

english_text = load_doc('traing_english.txt')
hindi_text = load_doc('traing_hindi.txt')
pairs = to_pairs(english_text, hindi_text)


# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,n_layers=1, bidirectional=True, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = int(hidden_size/2)
        self.hidden_dim = self.hidden_size
        self.num_layers = n_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, self.hidden_size, n_layers*2, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        fwd_final = hidden[0:hidden.size(0):2]
        bwd_final = hidden[1:hidden.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)
        return output, final

    def initHidden(self):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers*2,
            self.batch_size,
            self.hidden_size*2,
            device=device
        )


# In[ ]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.2, max_length=output_lang.n_words,n_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p/5
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn1 = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers*2)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        intmd_torch = torch.cat((embedded[0],hidden[0]),1)
        attn_out= self.attn1(intmd_torch)
        attn_weights = F.softmax(attn_out)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
       
        output = torch.cat((embedded, attn_applied), 0)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output = torch.squeeze(output,0)
        dec_hidden = self.initHidden()
        output1, dec_hidden = self.gru(output, hidden)
        output2 = F.log_softmax(self.out(output1[0]), dim=1)
        return output2, dec_hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1*2, 1, self.hidden_size, device=device)


# In[ ]:


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# In[ ]:


teacher_forcing_ratio = 1

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=output_lang.n_words):
    
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_dim*2, device=device)
    
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei])
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[ ]:


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[ ]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    random.shuffle(pairs)
    
    criterion = nn.NLLLoss()
    
    for iter in range(0, len(pairs)*40 + 1):
        training_pairs = tensorsFromPair(pairs[(iter)%len(pairs)])
        training_pair = training_pairs
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%.4f' % (print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# In[ ]:


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[ ]:


def evaluate(encoder, decoder, sentence, max_length=output_lang.n_words):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim*2, device=device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei])
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# In[ ]:


def evaluateRandomly(encoder, decoder, n=20):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[ ]:


embed_dim = 300
hidden_size = embed_dim
hidden_dim = embed_dim
n_layers = 1

encoder1 = EncoderRNN(input_lang.n_words, embed_dim, hidden_dim, n_layers).to(device)
attn_decoder1 = AttnDecoderRNN(embed_dim, output_lang.n_words, n_layers).to(device)


# In[ ]:


trainIters(encoder1, attn_decoder1, 75000, print_every=5000)


# In[ ]:


evaluateRandomly(encoder1, attn_decoder1)


# In[ ]:


def storetraintest(encoder, decoder, n):
    actual, predicted = list(), list()
    filename = 'sad'
    pairs1 = pairs
    if (n==1):
        filename = 'training_result_pytorch.txt'
    else:
        filename = 'testing_result_pytorch.txt'
        pairs1 = pairs_testing
    sample = open(filename, 'w')
    for i in range(len(pairs1)):
        pair = pairs1[i]
        print('Input: {}'.format(pair[0]), file = sample)
        print('Actual_translation: {}'.format(pair[1]), file = sample)
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('Predicted translation: {}'.format(output_sentence), file = sample)
        print ("", file=sample)
        
        hindi_list = pair[1].split()
        predicted_sen = output_sentence.split()
        if (hindi_list[-1] == "<EOS>"):
            hindi_list = hindi_list[:-1]
        if (hindi_list[0] == "<SOS>"):
            hindi_list.pop(0)
        if (predicted_sen[-1] == "<EOS>"):
            predicted_sen = predicted_sen[:-1]
        actual.append(hindi_list)
        predicted.append(predicted_sen)
        
    sample.close()
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
    print('Individual 1-gram: %f' % corpus_bleu(actual, predicted, weights=(1, 0, 0, 0)))
    print('Individual 2-gram: %f' % corpus_bleu(actual, predicted, weights=(0, 1, 0, 0)))
    print('Individual 3-gram: %f' % corpus_bleu(actual, predicted, weights=(0, 0, 1, 0)))
    print('Individual 4-gram: %f' % corpus_bleu(actual, predicted, weights=(0, 0, 0, 1)))


# In[ ]:


storetraintest(encoder1, attn_decoder1,1)


# In[ ]:


storetraintest(encoder1, attn_decoder1,0)

