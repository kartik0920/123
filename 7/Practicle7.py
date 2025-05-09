import pandas as pd
import nltk  # Natural Language Toolkit library widely used for NLP applications
import re  # Regular expressions used for pattern matching

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# ------------------------------------------------------------------------------------------------------------------------------
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

# ------------------------------------------------------------------------------------------------------------------------------
def computeIDF(documents):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        if val > 0:
            idfDict[word] = math.log(N / float(val))
        else:
            idfDict[word] = 0
    return idfDict

# ------------------------------------------------------------------------------------------------------------------------------
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

# ------------------------------------------------------------------------------------------------------------------------------
text = "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."
print('The given sentence is:\n', text)

# ------------------------------------------------------------------------------------------------------------------------------
# Sentence Tokenization
from nltk.tokenize import sent_tokenize
tokenized_text = sent_tokenize(text)
print("\nSentence Tokenization:\n", tokenized_text)

# Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)
print('\nWord Tokenization:\n', tokenized_word)

# ------------------------------------------------------------------------------------------------------------------------------
# POS Tagging
pos_tags = nltk.pos_tag(tokenized_word)
print("\nPart of Speech Tagging:\n", pos_tags)

# ------------------------------------------------------------------------------------------------------------------------------
# Removing Stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

text2 = "How to remove stop words with NLTK library in Python?"
text2 = re.sub('[^a-zA-Z]', ' ', text2)  # Removing non-alphabet characters
tokens = word_tokenize(text2.lower())
filtered_text = [w for w in tokens if w not in stop_words]

print("\nTokenized Sentence:\n", tokens)
print("Filtered Sentence (without stopwords):\n", filtered_text)

# ------------------------------------------------------------------------------------------------------------------------------
# Stemming
from nltk.stem import PorterStemmer
e_words = ["wait", "waiting", "waited", "waits"]
ps = PorterStemmer()
print("\nStemming Examples:")
for w in e_words:
    rootWord = ps.stem(w)
    print(f'Stemming for {w}: {rootWord}')

# ------------------------------------------------------------------------------------------------------------------------------
# Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text3 = "studies studying cries cry"
tokenization = nltk.word_tokenize(text3)

print("\nLemmatization Examples:")
for w in tokenization:
    print(f"Lemma for {w} is {wordnet_lemmatizer.lemmatize(w)}")
