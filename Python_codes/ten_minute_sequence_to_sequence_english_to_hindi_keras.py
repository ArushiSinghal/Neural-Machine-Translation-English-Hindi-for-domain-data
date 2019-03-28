#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array


# In[2]:


from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


# In[3]:


INDIC_NLP_LIB_HOME=r"/home/simran/NLP Applications/nla_project/indic_nlp_library"
INDIC_NLP_RESOURCES=r"/home/simran/NLP Applications/nla_project/indic_nlp_resources"


# In[4]:


import sys
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
sys.path.append(r'/home/simran/NLP Applications/nla_project/hindi-tokenizer')


# In[5]:


from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)


# In[6]:


from indicnlp import loader
loader.load()


# In[7]:


from indicnlp.normalize.indic_normalize import DevanagariNormalizer
factory = DevanagariNormalizer()


# In[8]:


from indicnlp.tokenize import indic_tokenize  


# In[9]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# In[10]:


text = load_doc('data/eng-hindi.txt')


# In[11]:


#text


# In[12]:


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs


# In[13]:


pairs = to_pairs(text)
#pairs


# In[14]:


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
    text=text.replace(u"‘",'')
    text=text.replace(u"+",'')
    text=text.replace(u"%",'')
    text=text.replace(u"…",'')
    text=text.replace(u"”",'')
    text=text.replace(u"\\",'')
    text=text.replace(u"_",'')
    text=text.replace(u"[",'')
    text=text.replace(u"]",'')
    text=re.sub('[a-zA-Z]', '', text)
    text=re.sub('[0-9+\-*/.%>=!;~{}×–`’]', '', text)
    return text


# In[15]:


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        
        line = pair[0]
        #print (line)
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = line.split()
        line = [word.lower() for word in line]
        line = [word.translate(table) for word in line]
        line = [re_print.sub('', w) for w in line]
        line = [word for word in line if word.isalpha()]
        clean_pair.append(' '.join(line))
        
        line = pair[1]
        #print (line)
        line = factory.normalize(line)
        line = clean_text(line)
        tokens = list()
        for t in indic_tokenize.trivial_tokenize(line): 
            tokens.append(t)
        line = tokens
        line = [word.lower() for word in line]
        line = [word for word in line if not re.search(r'\d', word)]
        clean_pair.append(' '.join(line))
        
        print (clean_pair)
        
        cleaned.append(clean_pair)
    return array(cleaned)


# In[16]:


clean_pairs = clean_pairs(pairs)
clean_pairs


# In[17]:


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# In[18]:


save_clean_data(clean_pairs, 'clean_english-hindi.pkl')


# In[19]:


# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))


# In[20]:


len(clean_pairs)


# In[21]:


print('[%s] => [%s]' % (clean_pairs[24999,0], clean_pairs[24999,1]))


# # Code of ten minute sequence starts from here

# In[22]:


lines = clean_pairs
lines = lines[:45000, :]
len(lines)


# In[23]:


num_samples = 50000
num_samples


# In[ ]:


input_texts = []
target_texts = []
input_characters = set()
target_characters = set()


# In[ ]:


for line in lines:
    temp = line[0] + "\t" + line[1]
    #print (a)
    line = temp
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


# In[ ]:


print (len(input_texts))


# In[ ]:


print (len(target_texts))


# In[ ]:


input_characters


# In[ ]:


target_characters


# In[ ]:


input_texts[155]


# In[ ]:


target_texts[155]


# In[ ]:


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[ ]:


print(input_characters)


# In[ ]:


print(target_characters)


# In[ ]:


input_token_index = dict(
  [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
  [(char, i) for i, char in enumerate(target_characters)])


# In[ ]:


import numpy as np

encoder_input_data = np.zeros(
  (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
  dtype='float32')
decoder_input_data = np.zeros(
  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')
decoder_target_data = np.zeros(
  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')


# In[ ]:


encoder_input_data.shape


# In[ ]:


decoder_input_data.shape


# In[ ]:


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# In[ ]:


encoder_input_data[155].shape


# In[ ]:


import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


# In[ ]:


batch_size = 64  # batch size for training
epochs = 100  # number of epochs to train for
latent_dim = 256  # latent dimensionality of the encoding space


# In[ ]:


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]


# In[ ]:


decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[ ]:


model = Model(inputs=[encoder_inputs, decoder_inputs], 
              outputs=decoder_outputs)


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()


# In[ ]:


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('seq2seq_eng-hindi.h5')


# In[ ]:


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights('seq2seq_eng-hindi.h5')


# In[ ]:


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
  decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states)


# In[ ]:


# reverse-lookup token index to turn sequences back to characters
reverse_input_char_index = dict(
  (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
  (i, char) for char, i in target_token_index.items())


# In[ ]:


def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
              stop_condition = True
      

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
    
        states_value = [h, c]
    
    return decoded_sentence


# In[ ]:


for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# In[ ]:


input_sentence = "How are you?"
test_sentence_tokenized = np.zeros(
  (1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
for t, char in enumerate(input_sentence):
    test_sentence_tokenized[0, t, input_token_index[char]] = 1.
print(input_sentence)
print(decode_sequence(test_sentence_tokenized))


# In[ ]:


val_input_texts = []
val_target_texts = []
line_ix = 46000
for line in lines[line_ix:line_ix+10]:
    input_text, target_text = line.split('\t')
    val_input_texts.append(input_text)
    val_target_texts.append(target_text)

val_encoder_input_data = np.zeros(
  (len(val_input_texts), max([len(txt) for txt in val_input_texts]),
   num_encoder_tokens), dtype='float32')

for i, input_text in enumerate(val_input_texts):
    for t, char in enumerate(input_text):
        val_encoder_input_data[i, t, input_token_index[char]] = 1.


# In[ ]:


for seq_index in range(10):
    input_seq = val_encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', val_input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence[:-1])
    print('Ground Truth sentence:', val_target_texts[seq_index])


# In[ ]:


max([len(txt) for txt in val_input_texts])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #from pickle import load
# from pickle import dump
# from numpy.random import rand
# from numpy.random import shuffle
# from numpy import array
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.utils.vis_utils import plot_model
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Embedding
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras.callbacks import ModelCheckpoint
# 
# #load a clean dataset
# def load_clean_sentences(filename):
#     return load(open(filename, 'rb'))
# 
# #save a list of clean sentences to file
# def save_clean_data(sentences, filename):
#     dump(sentences, open(filename, 'wb'))
#     print('Saved: %s' % filename)
# 
# #load dataset
# raw_dataset = load_clean_sentences('clean_english-hindi.pkl')
# 
# #reduce dataset size
# 
# '''
# delete the lower code later
# '''
# 
# n_sentences = 10000
# dataset = raw_dataset[:n_sentences, :]
# 
# train, test = dataset[:9000], dataset[9000:]
# 
# '''
# Originally it should be like this
# n_sentences = 49999
# dataset = raw_dataset[:n_sentences, :]
# 
# train, test = dataset[:25000], dataset[25000:]
# '''
# 
# shuffle(train)
# shuffle(test)
# shuffle(dataset)
# 
# print(len(train))
# print(len(test))
# 
# save_clean_data(dataset, 'clean_english-hindi-both.pkl')
# save_clean_data(train, 'clean_english-hindi-train.pkl')
# save_clean_data(test, 'clean_english-hindi-test.pkl')
#load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))
 
#load datasets
dataset = load_clean_sentences('clean_english-hindi-both.pkl')
train = load_clean_sentences('clean_english-hindi-train.pkl')
test = load_clean_sentences('clean_english-hindi-test.pkl')def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizerdef max_length(lines):
    return max(len(line.split()) for line in lines)
# In[ ]:


# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
eng_tokenizer


# In[ ]:


print(eng_tokenizer.document_count)
print(eng_tokenizer.word_counts)
print(eng_tokenizer.word_index)
print(eng_tokenizer.word_docs)


# #prepare german tokenizer
# hindi_tokenizer = create_tokenizer(dataset[:, 1])
# hindi_vocab_size = len(hindi_tokenizer.word_index) + 1
# hindi_length = max_length(dataset[:, 1])
# print('Hindi Vocabulary Size: %d' % hindi_vocab_size)
# print('Hindi Max Length: %d' % (hindi_length))

# In[ ]:


print(hindi_tokenizer.document_count)
print(hindi_tokenizer.word_counts)
print(hindi_tokenizer.word_index)
print(hindi_tokenizer.word_docs)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X# one hot encode target sequence
import numpy

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        #print (encoded)
        ylist.append(encoded)
    y = array(ylist, dtype=numpy.uint8)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y
# In[ ]:


trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainX


# In[ ]:


trainY = encode_sequences(hindi_tokenizer, hindi_length, train[:, 1])
trainY


# In[ ]:


trainY = encode_output(trainY, hindi_vocab_size)
trainY


# In[ ]:


testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testX


# In[ ]:


testY = encode_sequences(hindi_tokenizer, hindi_length, test[:, 1])
testY


# In[ ]:


testY = encode_output(testY, hindi_vocab_size)
testY

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
# model = define_model(eng_vocab_size, hindi_vocab_size, eng_length, hindi_length, 256)

# model.compile(optimizer='adam', loss='categorical_crossentropy')

# In[ ]:


#from keras.utils.vis_utils import plot_model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)


# In[ ]:


# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)


# In[ ]:


dataset = load_clean_sentences('clean_english-hindi-both.pkl')
train = load_clean_sentences('clean_english-hindi-train.pkl')
test = load_clean_sentences('clean_english-hindi-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

# prepare german tokenizer
hindi_tokenizer = create_tokenizer(dataset[:, 1])
hindi_vocab_size = len(hindi_tokenizer.word_index) + 1
hindi_length = max_length(dataset[:, 1])

# prepare data
trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])


# In[ ]:


# load model
model = load_model('model.h5')


# In[ ]:





# In[ ]:


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[ ]:


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# In[ ]:


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# In[ ]:


# load model
model = load_model('model.h5')
# test on some training sequences
print('train')
evaluate_model(model, hindi_tokenizer, trainX, train)
# test on some test sequences
print('test')
evaluate_model(model, hindi_tokenizer, testX, test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


INDIC_NLP_LIB_HOME


# In[ ]:


INDIC_NLP_RESOURCES


# In[ ]:


for path in sys.path:
    print (path)


# In[ ]:


from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

input_text="\u0958 \u0915\u093c"
remove_nuktas=False
factory=IndicNormalizerFactory()
normalizer=factory.get_normalizer("hi",remove_nuktas)
output_text=normalizer.normalize(input_text)

print(output_text)
print('Length before normalization: {}'.format(len(input_text)))
print('Length after normalization: {}'.format(len(output_text)))


# In[ ]:


from indicnlp.normalize.indic_normalize import DevanagariNormalizer
input_text = "अत : इसे बिना टाँके वाला ऑपरेशन भी कहते हैं ।"
factory1=DevanagariNormalizer()
#normalizer1=factory1.get_normalizer("hi",remove_nuktas)
output_text1=factory1.normalize(input_text)
some = factory1.get_char_stats(input_text)
print (some)
print (input_text)
print(output_text1)
print('Length before normalization: {}'.format(len(input_text)))
print('Length after normalization: {}'.format(len(output_text1)))


# In[ ]:


from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
input_text='राजस्थान'
# input_text='രാജസ്ഥാന'
# input_text='රාජස්ථාන'
print(UnicodeIndicTransliterator.transliterate(input_text,"hi","mr"))


# In[ ]:


from indicnlp.tokenize import indic_tokenize  

indic_string='अनूप,अनूप?।फोन,'

print('Input String: {}'.format(indic_string))
print('Tokens: ')
a = list()
for t in indic_tokenize.trivial_tokenize(indic_string): 
    a.append(t)
print (a)


# In[ ]:





# In[ ]:


from HindiTokenizer import Tokenizer


# In[ ]:


t=Tokenizer("यह वाक्य हिन्दी में है।")


# In[ ]:




