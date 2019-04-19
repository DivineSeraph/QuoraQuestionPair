import re
import io
from string import punctuation
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.models import load_model
import keras.backend as K

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pickle


########################################
# set directories and parameters
########################################
BASE_DIR = os.getcwd() + '/input/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
MAX_WORDS = 200000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 300


stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

def text_to_wordlist(text, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally remove stop words
    if remove_stopwords:
        text = [w for w in text if not w in stop_words]
        
    text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    return(text)

question1 = input('Enter question 1: ')
question2 = input('Enter question 2: ')

print('\nThe question pair entered is\n {0} \n {1}'.format(question1,question2))

question1 = text_to_wordlist(question1)
question2 = text_to_wordlist(question2)

print('\nThe cleaned question pair entered is\n {0} \n {1}'.format(question1,question2))

question1 = [question1]
question2 = [question2]

print(question1)
print(question2)
#load tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences_1 = tokenizer.texts_to_sequences(question1)
sequences_2 = tokenizer.texts_to_sequences(question2)
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

print(sequences_1) 
print(sequences_2)


'''word_index = tokenizer.word_index
print('\nFound %s unique tokens' % len(word_index))'''

######## Prepare word embeddings #########
'''print('\nIndexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

print("Preparing embedding Matrix")
nb_words = min(MAX_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in tqdm(word_index.items()):
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))'''


#np.save('embedding_weights',embedding_matrix)

embedding_matrix = np.load('embedding_weights.npy')
#embedding_matrix.dump('embedding_weights.dat')
##########################################

########## Model Building ################
n_hidden = 50
batch_size = 64
n_epoch = 6

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
left_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
right_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(embedding_matrix),EMBEDDING_DIM, weights = [embedding_matrix], input_length = MAX_SEQUENCE_LENGTH, trainable = False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# load weights
malstm.load_weights("weights.best.hdf5")
# load weights into new model
#loaded_model.load_weights("weights.best.h5")
print("Loaded model from disk")

#print(malstm.summary())
output = malstm.predict([data_1,data_2])

print('Percentage of similarity is ',output)

print(output>0.5)




