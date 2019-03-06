########################################
## import packages
########################################
import os
import pandas as pd

import re
from string import punctuation
from nltk.corpus import stopwords

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
test = pd.read_csv(TEST_DATA_FILE)

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
    for question in questions:

        question_list.append(text_to_wordlist(question))

        if(len(question_list) % 100000 == 0):
            # Progress is reported every 100000 questions
            cur_progress = (len(question_list)/list_len)*100

            print("{n} is {p}% complete".format(
                n=list_name, p=round(cur_progress, 2)))


########################################


print('Processing text dataset')

#########   Noise removal   ############
print('Null values in train.csv')
print(train.isnull().sum())
print('\nNull values in test.csv')
print(test.isnull().sum())

print('\nDropping null values\n')
train = train.dropna()
test = test.dropna()

print('Null values in train.csv')
print(train.isnull().sum())
print('\nNull values in test.csv')
print(test.isnull().sum())

print(train.head(5))
print('\n')
train_question1 = []
clean_questions(train_question1,train.question1, "Train: Question 1 List", len(train))

train_question2 = []
clean_questions(train_question2,train.question2, "Train: Question 2 List",len(train))


for i in range(5):
    print("{q1}\n{q2}".format(q1=train_question1[i],q2=train_question2[i]))
    print('\n\n')

########################################

############ Tokenizer #################
