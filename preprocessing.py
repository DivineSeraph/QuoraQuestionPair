########################################
## import packages
########################################
import os
import pandas as pd


from nltk.corpus import stopwords

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


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]


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

########################################

############ Tokenizer #################
