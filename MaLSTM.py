########################################
## import packages
########################################
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import re
from string import punctuation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

########################################
# set directories and parameters
########################################
BASE_DIR = os.getcwd() + '/input/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'


########################################
# process texts in datasets
########################################

train = pd.read_csv(TRAIN_DATA_FILE)
MAX_WORDS = 200000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 300
#test = pd.read_csv(TEST_DATA_FILE)

##########   Functions   ###############


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


def clean_questions(question_list, questions, list_name, list_len):
    # Each question is cleaned using text_to_wordlist and append to a list
    for question in tqdm(questions):

        question_list.append(text_to_wordlist(question))

        #if(len(question_list) % 100000 == 0):
            # Progress is reported every 100000 questions
         #   cur_progress = (len(question_list)/list_len)*100

          #  print("{n} is {p}% complete".format(
           #     n=list_name, p=round(cur_progress, 2)))


########################################


print('Processing text dataset')

#########   Noise removal   ############
print('Null values in train.csv')
print(train.isnull().sum())
#print('\nNull values in test.csv')
#print(test.isnull().sum())

print('\nDropping null values\n')
train = train.dropna()
#test = test.dropna()

print('Null values in train.csv')
print(train.isnull().sum())
#print('\nNull values in test.csv')
#print(test.isnull().sum())

print(train.head(5))
print('\n')
print('Cleaning question 1 coloumn:')
train_question1 = []
clean_questions(train_question1,train.question1, "Train: Question 1 List", len(train))
train['question1'] = train_question1
print('\n')
print('Cleaning question 2 coloumn:')
train_question2 = []
clean_questions(train_question2,train.question2, "Train: Question 2 List",len(train))
train['question2'] = train_question2
print('\n')
print(train.head(5))

########################################
train_size = int(len(train)* .8)
print("\nTrain Size = %d" % train_size)
print("\nTest size = %d" % (len(train)-train_size))
question1_train = train['question1'].values[:train_size]
question2_train = train['question2'].values[:train_size]
isDuplicate_train = train['is_duplicate'].values[:train_size]

question1_validate = train['question1'].values[train_size:]
question2_validate = train['question2'].values[train_size:]
isDuplicate_validate = train['is_duplicate'].values[train_size:]

print(question1_train[:5])
print('\n')
print(question2_train[:5])
########################################

############ Tokenizer #################

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(list(question1_train)+list(question2_train)+list(question1_validate)+list(question2_validate))

sequences_1 = tokenizer.texts_to_sequences(question1_train)
sequences_2 = tokenizer.texts_to_sequences(question2_train)
test_sequences_1 = tokenizer.texts_to_sequences(question1_validate)
test_sequences_2 = tokenizer.texts_to_sequences(question2_validate)

word_index = tokenizer.word_index
print('\nFound %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(isDuplicate_train)
print('\nShape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_labels = np.array(isDuplicate_validate)
print('\nShape of data tensor:', test_data_1.shape)
print('Shape of label tensor:', test_labels.shape)

#saving the created tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('\nTokenizer saved in tokenizer.pickle')
##########################################

######## Prepare word embeddings #########
print('\nIndexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

print("Preparing embedding Matrix")
nb_words = min(MAX_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in tqdm(word_index.items()):
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#np.save('embedding_weights',embedding_matrix)

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
#malstm.load_weights("weights.best.hdf5")

#compile model
malstm.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

# checkpoint for best weight
#filepath="weights.best.hdf5"

# checkpoint for each iteration
filepath="data/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

malstm_trained = malstm.fit([data_1, data_2], labels, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([test_data_1, test_data_2], test_labels),callbacks=callbacks_list)

# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = malstm.to_json()

with open("model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
#malstm_trained.save_weights("model_num.h5")