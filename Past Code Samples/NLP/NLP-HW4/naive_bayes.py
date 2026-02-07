"""
Names: Maitrayee Deka (deka0031), Monika Bhardwaj (bhard046)

Report:
Your results (precision, recall, and F-Measure)
-- Precision is  0.42990654205607476
-- Recall is  0.23115577889447236
-- F-Measure is  0.3006535947712418
How you handled the tokens (i.e. what did you ignore, if anything?)
-- Removed puctuations, stop-words
-- Ignored words in test corpus that are not in Vocab.
What smoothing did you use?
-- Laplace (add one)
Did you add any other tricks (i.e. negation-handling, etc.)?
-- No
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import numpy as np
import math

corpus = pd.DataFrame() #datastructure used is pandas dataframes

def createMovieReviewsDF(corpus):
    #Note: this function loads all the files from the movie_reviews folder to create
    #      a pandas dataframe out of it (easier to handle/manipulate)

    pos_path = "movie_reviews/pos"
    neg_path = "movie_reviews/neg"
    for file in os.listdir(pos_path):
        file_path = pos_path + '/' + file
        with open(file_path, mode="r") as f:
            inputString = f.read()
            corpus = corpus.append({'ID': file, 'Text': inputString, 'Sentiment': 'pos'}, ignore_index = True)
    for file in os.listdir(neg_path):
        file_path = neg_path + '/' + file
        with open(file_path, mode="r") as f:
            inputString = f.read()
            corpus = corpus.append({'ID': file, 'Text': inputString, 'Sentiment': 'neg'}, ignore_index = True)
    return(corpus)


def splitTrainTest(df):
    # Split the data into train and test (doing an 80/20 split here with random seed)
    trainText, testText = train_test_split(df, test_size = 0.2, random_state = 42)
    return (trainText, testText)


def preprocessText(df):
    #Removing punctuation
    print("Preprocessing text: removing punctuation...")
    df['Text'] = df['Text'].str.replace(r'[^\w\s]+', '', regex=True)
    return(df)



def countFrequency(dict, inputSring):
    #Creates a dictionary with word frequencies
    words = inputSring.split()

    for word in words:
        if word in dict.keys():
            dict[word] = dict[word] + 1
        else:
            dict[word] = 1

    return(dict)

def createBagOfWords(trainCorpus):

    print('Creating bag of words')
    stopwords_english = stopwords.words("english")
    #For both the negative and positive classes, create bag of words
    pos_counts = {}
    neg_counts = {}
    pos_df = trainCorpus.loc[(trainCorpus['Sentiment'] == 'pos')]
    neg_df = trainCorpus.loc[(trainCorpus['Sentiment'] == 'neg')]

    positive_text = ' '.join(pos_df['Text'].tolist()).lower()
    clean_positive_text = ' '.join([word for word in positive_text.split() if word not in stopwords_english])
    pos_counts = countFrequency(pos_counts, clean_positive_text)

    negative_text = ' '.join(neg_df['Text'].tolist()).lower()
    clean_negative_text = ' '.join([word for word in negative_text.split() if word not in stopwords_english])
    neg_counts = countFrequency(neg_counts, clean_negative_text)

    return(pos_counts, neg_counts)



def combineDictionaries(dict1, dict2):
    combinedDict = dict1.copy()
    combinedDict.update(dict2)
    return(combinedDict)

def calculateScores(classifiedDf):
    truePositives, trueNegatives, falsePositives, falseNegatives = 0, 0, 0, 0
    precision, recall, f_measure = 0.0, 0.0, 0.0

    for i in range(0, len(classifiedDf)):
        sentiment = classifiedDf.iloc[i]['Sentiment']
        classification = classifiedDf.iloc[i]['Classification']

        if sentiment == 'pos' and classification == 'pos':
            truePositives +=1

        elif sentiment == 'pos' and classification == 'neg':
            falseNegatives += 1

        elif sentiment == 'neg' and classification == 'neg':
            trueNegatives += 1

        elif sentiment == 'neg' and classification == 'pos':
            falsePositives += 1

        else:
            pass

    precision = truePositives/(truePositives + falsePositives)
    recall = truePositives/(truePositives + falseNegatives)
    f_measure = (2*precision*recall) / (precision + recall)

    return (precision, recall, f_measure)



def classifyTest(testCorpus, pos_counts, neg_counts, pos_prob, neg_prob):
    #Classify the test corpus
    stopwords_english = stopwords.words("english")
    testCorpus["Classification"] = np.nan
    num_distinct_pos = len(list(pos_counts.keys())) #number of distinct positive words
    num_distinct_neg = len(list(neg_counts.keys())) #umber of distinct negative words
    vocab_dict = combineDictionaries(pos_counts, neg_counts)
    vocab_size = len(vocab_dict) #number of unique negative and positive words, combined

    testCorpus = testCorpus.reset_index(drop=True)
    print('Test under progress.')
    for i in range(0, len(testCorpus)):
        #Count the sentence probability in this for loop (negative and positive)
        sentence_pos_probability = math.log(pos_prob, 10) #add positive class probability to sentence probability
        sentence_neg_probability = math.log(neg_prob, 10) #add negative class probability to sentence probability
        rowText = testCorpus.iloc[i]['Text']
        clean_rowText = ' '.join([word for word in rowText.split() if word not in stopwords_english])
        for word in clean_rowText:
            if vocab_dict.get(word):
                if word in pos_counts.keys():
                    word_prob_pos = (pos_counts[word] + 1)/(num_distinct_pos + vocab_size)
                else:
                    word_prob_pos = (1)/(num_distinct_pos + vocab_size)
                sentence_pos_probability = sentence_pos_probability + math.log(word_prob_pos, 10)

                if word in neg_counts.keys():
                    word_prob_neg = (neg_counts[word] + 1)/(num_distinct_neg + vocab_size)
                else:
                    word_prob_neg = (1)/(num_distinct_neg + vocab_size)
                sentence_neg_probability = sentence_neg_probability + math.log(word_prob_neg, 10)

        if (sentence_pos_probability > sentence_neg_probability):
            testCorpus.loc[testCorpus.index[i], 'Classification'] = "pos"

        elif (sentence_pos_probability < sentence_neg_probability):
            testCorpus.loc[testCorpus.index[i], 'Classification'] = "neg"


    precision, recall, f_measure = calculateScores(testCorpus)

    print("Precision is ", precision)
    print("Recall is ", recall)
    print("F-Measure is ", f_measure)



def main():
    df = createMovieReviewsDF(corpus)


    preProcessDf = preprocessText(df)
    trainCorpus, testCorpus = splitTrainTest(preProcessDf)

    numDocs = len(trainCorpus)
    prob_pos = len(trainCorpus.loc[(trainCorpus['Sentiment'] == 'pos')])/numDocs
    prob_neg = len(trainCorpus.loc[(trainCorpus['Sentiment'] == 'neg')])/numDocs


    pos_counts, neg_counts = createBagOfWords(trainCorpus)

    classifyTest(testCorpus, pos_counts, neg_counts, prob_pos, prob_neg)


if __name__ == '__main__':
   main()
