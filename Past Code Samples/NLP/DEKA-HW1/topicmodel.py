import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import itertools
import collections
import string


def mostFrequentWords(wordList, n):
    counts = collections.Counter(wordList)
    return(counts.most_common(n))


def topicWords(lines):
    allLines = ''.join(lines)
    stopWords = stopwords.words("english")
    wordTokens = word_tokenize(allLines)
    cleanText = []
    punctuations = string.punctuation

    for word in wordTokens:
        if (word in stopwords.words() or word in punctuations):
            pass
        else:
            cleanText.append(word)

    frequentWords = mostFrequentWords(cleanText, 5)

    for word in frequentWords:
        print(word[0])


def main():
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        topicWords(lines)


if __name__ == '__main__':
   main()
