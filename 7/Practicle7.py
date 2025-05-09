# Step 1 & 2: Import Libraries and Initialize Text
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math


text = "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."

# Step 3: Tokenization
# Sentence Tokenization
tokenized_text = sent_tokenize(text)
print("Sentence Tokenization:\n", tokenized_text)

# Word Tokenization
tokenized_word = word_tokenize(text)
print("\nWord Tokenization:\n", tokenized_word)

# Step 4: Remove Punctuations & Stop Words
text_clean = "How to remove stop words with NLTK library in Python?"
text_clean = re.sub('[^a-zA-Z]', ' ', text_clean)
tokens = word_tokenize(text_clean.lower())

stop_words = set(stopwords.words("english"))
filtered_text = [w for w in tokens if w not in stop_words]

print("\nTokenized Sentence:", tokens)
print("Filtered Sentence:", filtered_text)

# Step 5: Stemming
e_words = ["wait", "waiting", "waited", "waits"]
ps = PorterStemmer()
print("\nStemming Results:")
for w in e_words:
    print(ps.stem(w))

# Step 6: Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
text_lemma = "studies studying cries cry"
tokenization = word_tokenize(text_lemma)
print("\nLemmatization Results:")
for w in tokenization:
    print(f"Lemma for {w} is {wordnet_lemmatizer.lemmatize(w)}")

# Step 7: POS Tagging
data = "The pink sweater fit her perfectly"
words = word_tokenize(data)
print("\nPOS Tagging:")
for word in words:
    print(nltk.pos_tag([word]))

# -----------------------------
# TF-IDF Manual Computation
# -----------------------------
# Step 1: Documents
documentA = 'Jupiter is the largest Planet'
documentB = 'Mars is the fourth planet from the Sun'

# Step 2: Bag of Words
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

# Step 3: Unique Words
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

# Step 4: Word Count Dictionary
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1

numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

# Step 5: Compute TF
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)

# Step 6: Compute IDF
def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

idfs = computeIDF([numOfWordsA, numOfWordsB])
print("\nIDF Values:\n", idfs)

# Step 7: Compute TF-IDF
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

df = pd.DataFrame([tfidfA, tfidfB])
print("\nTF-IDF Table:\n", df)

# -----------------------------
# Bonus: TF-IDF with Sklearn
# -----------------------------
corpus = [documentA, documentB]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("\nTF-IDF using Sklearn:")
print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()))
