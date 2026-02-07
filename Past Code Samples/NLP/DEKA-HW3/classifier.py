
############################ AUTHOR: Maitrayee Deka ############################

#INSTRUCTIONS
# To run the program without a test file and automatically extract train/dev sets
#run as python classifier.py authorlist.txt where author list contains the corpus list
##WARNING! - in order for the program to run correctly, please provide the authorlist text
# in the order: austen_utf8.txt, dickens_utf8.txt, tolstoy_utf8.txt, wilde_utf8.txt

# When the program is run with a test flag, it will use the provided test file as the test set

# What encoding type does your program run on?
# Answer: This program runs on the UTF-8 encoding

# What information is in your Language Models (bigrams, trigrams, etc)?
# Answer: I used trigrams.

# What method of smoothing are you using?
# Answer: I am using Laplace smoothing for all my models.

# Any other tweaks you made to improve results (backoff, etc.)
# Answer: I used regular expressions to preprocess the text and remove punctuations.

# The results you get with the given data with an automatically-extracted development set (i.e. the output from running it without the -test flag)
# Results on dev set:
# austen       83.70927318295739 % correct
# dickens       40.98360655737705 % correct
# tolstoy       20.0 % correct
# wilde       91.01861993428258 % correct

################################################################################


import sys
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace
import math


def preprocess(inputString):
    cleanInput =  re.sub(r'[^\w\s]', '', inputString)
    return cleanInput.lower()


def processFile(fileName):
    with open(fileName, mode="r", encoding="utf-8") as f:
        inputString = f.read().lstrip("\ufeff")
    return inputString

def extractDevSet(sentences):
    trainText, testText = train_test_split(sentences, test_size = 0.2, random_state = 42)
    return (trainText, testText)

def calculateMinPerplexityPercentage(scores_austen, scores_dickens, scores_tol, scores_wilde, ground_truth):

    correct = 0

    for i in range(0, len(scores_austen)):
        perplexities = [scores_austen[i], scores_dickens[i], scores_tol[i], scores_wilde[i]]

        minPerplexity = min(perplexities)
        if (minPerplexity == scores_austen[i] and ground_truth == 'austen'):
            correct += 1
        elif (minPerplexity == scores_dickens[i] and ground_truth == 'dickens'):
            correct += 1
        elif (minPerplexity == scores_tol[i] and ground_truth == 'tolstoy'):
            correct += 1
        elif (minPerplexity == scores_wilde[i] and ground_truth == 'wilde'):
            correct += 1

    return (correct/len(scores_austen) * 100)



def calculateSentenceAuthor(scores_austen, scores_dickens, scores_tol, scores_wilde):
    for i in range(0, len(scores_austen)):
        perplexities = [scores_austen[i], scores_dickens[i], scores_tol[i], scores_wilde[i]]
        minPerplexity = min(perplexities)

        if (minPerplexity == scores_austen[i]):
            print('austen')
        elif (minPerplexity == scores_dickens[i]):
            print('dickens')
        elif (minPerplexity == scores_tol[i]):
            print('tolstoy')
        elif (minPerplexity == scores_wilde[i]):
            print('wilde')
    return




def buildLM(fileNames, extractDev):
    n = 3

    authors = ['austen', 'dickens', 'tolstoy', 'wilde']


    #Tokenization All
    tokenizedSentencesTrainingAusten = []
    tokenizedSentencesTestAusten = []
    inputStringAusten = processFile(fileNames[0])
    cleanTextAusten = re.sub(r"\n", '', inputStringAusten)
    sentencesAusten = nltk.sent_tokenize(cleanTextAusten)

    tokenizedSentencesTrainingDickens = []
    tokenizedSentencesTestDickens = []
    inputStringDickens = processFile(fileNames[1])
    cleanTextDickens = re.sub(r"\n", '', inputStringDickens)
    sentencesDickens = nltk.sent_tokenize(cleanTextDickens)

    tokenizedSentencesTrainingTol = []
    tokenizedSentencesTestTol = []
    inputStringTol = processFile(fileNames[2])
    cleanTextTol = re.sub(r"\n", '', inputStringTol)
    sentencesTol = nltk.sent_tokenize(cleanTextTol)

    tokenizedSentencesTrainingWilde = []
    tokenizedSentencesTestWilde = []
    inputStringWilde = processFile(fileNames[3])
    cleanTextWilde = re.sub(r"\n", '', inputStringWilde)
    sentencesWilde = nltk.sent_tokenize(cleanTextWilde)
    ############################################################################


    #EXTRACT TRAIN AND DEV

    if extractDev:
        print("Splitting into training and development...")
        trainCorpusAusten, developmentSetAusten = extractDevSet(sentencesAusten)
        trainCorpusDickens, developmentSetDickens = extractDevSet(sentencesDickens)
        trainCorpusTol, developmentSetTol = extractDevSet(sentencesTol)
        trainCorpusWilde, developmentSetWilde = extractDevSet(sentencesWilde)

    else:
        trainCorpusAusten = sentencesAusten
        trainCorpusDickens = sentencesDickens
        trainCorpusTol = sentencesTol
        trainCorpusWilde = sentencesWilde


    #Training all LM's
    print("Training LMs...")
    #AUSTEN LM
    for sentence in trainCorpusAusten:
        cleanSentence = preprocess(sentence)
        tokens = nltk.word_tokenize(cleanSentence)
        tokenizedSentencesTrainingAusten.append(tokens)
    train_dataAusten, vocabAusten = padded_everygram_pipeline(n, tokenizedSentencesTrainingAusten)
    languageModelAusten = Laplace(n)
    languageModelAusten.fit(train_dataAusten, vocabAusten)

    #DICKENS LM
    for sentence in trainCorpusDickens:
        cleanSentence = preprocess(sentence)
        tokens = nltk.word_tokenize(cleanSentence)
        tokenizedSentencesTrainingDickens.append(tokens)
    train_dataDickens, vocabDickens = padded_everygram_pipeline(n, tokenizedSentencesTrainingDickens)
    languageModelDickens = Laplace(n)
    languageModelDickens.fit(train_dataDickens, vocabDickens)

    #TOLSTOY LM
    for sentence in trainCorpusTol:
        cleanSentence = preprocess(sentence)
        tokens = nltk.word_tokenize(cleanSentence)
        tokenizedSentencesTrainingTol.append(tokens)
    train_dataTol, vocabTol = padded_everygram_pipeline(n, tokenizedSentencesTrainingTol)
    languageModelTol = Laplace(n)
    languageModelTol.fit(train_dataTol, vocabTol)

    #WILDE LM
    for sentence in trainCorpusWilde:
        cleanSentence = preprocess(sentence)
        tokens = nltk.word_tokenize(cleanSentence)
        tokenizedSentencesTrainingWilde.append(tokens)
    train_dataWilde, vocabWilde = padded_everygram_pipeline(n, tokenizedSentencesTrainingWilde)
    languageModelWilde = Laplace(n)
    languageModelWilde.fit(train_dataWilde, vocabWilde)
    ############################################################################


    if extractDev:
        #TESTING AUSTEN
        austenlm_austentest = []
        dickenslm_austentest = []
        tolstoylm_austentest = []
        wildelm_austentest = []

        tokenizedSentencesTestFile = []
        for sentence in developmentSetAusten:
            cleanSentence = preprocess(sentence)
            tokens = nltk.word_tokenize(cleanSentence)
            tokenizedSentencesTestFile.append(tokens)

        for author in authors:
            test_austen, _ = padded_everygram_pipeline(n, tokenizedSentencesTestFile)
            for test in test_austen:
                if author == 'austen':
                    austenlm_austentest.append(languageModelAusten.perplexity(test))
                elif author == 'dickens':
                    dickenslm_austentest.append(languageModelDickens.perplexity(test))
                elif author == 'tolstoy':
                    tolstoylm_austentest.append(languageModelTol.perplexity(test))
                elif author == 'wilde':
                    wildelm_austentest.append(languageModelWilde.perplexity(test))

        austenPercentage = calculateMinPerplexityPercentage(austenlm_austentest, dickenslm_austentest, tolstoylm_austentest, wildelm_austentest, 'austen')

        #TESTING DICKENS
        austenlm_dickenstest = []
        dickenslm_dickenstest = []
        tolstoylm_dickenstest = []
        wildelm_dickenstest = []

        tokenizedSentencesTestFile = []
        for sentence in developmentSetDickens:
            cleanSentence = preprocess(sentence)
            tokens = nltk.word_tokenize(cleanSentence)
            tokenizedSentencesTestFile.append(tokens)

        for author in authors:
            test_dickens, _ = padded_everygram_pipeline(n, tokenizedSentencesTestFile)
            for test in test_dickens:
                if author == 'austen':
                    austenlm_dickenstest.append(languageModelAusten.perplexity(test))
                elif author == 'dickens':
                    dickenslm_dickenstest.append(languageModelDickens.perplexity(test))
                elif author == 'tolstoy':
                    tolstoylm_dickenstest.append(languageModelTol.perplexity(test))
                elif author == 'wilde':
                    wildelm_dickenstest.append(languageModelWilde.perplexity(test))

        dickensPercentage = calculateMinPerplexityPercentage(austenlm_dickenstest, dickenslm_dickenstest, tolstoylm_dickenstest, wildelm_dickenstest, 'dickens')

        #TESTING TOLSTOY
        austenlm_toltest = []
        dickenslm_toltest = []
        tolstoylm_toltest = []
        wildelm_toltest = []

        tokenizedSentencesTestFile = []
        for sentence in developmentSetTol:
            cleanSentence = preprocess(sentence)
            tokens = nltk.word_tokenize(cleanSentence)
            tokenizedSentencesTestFile.append(tokens)

        for author in authors:
            test_tol, _ = padded_everygram_pipeline(n, tokenizedSentencesTestFile)
            for test in test_tol:
                if author == 'austen':
                    austenlm_toltest.append(languageModelAusten.perplexity(test))
                elif author == 'dickens':
                    dickenslm_toltest.append(languageModelDickens.perplexity(test))
                elif author == 'tolstoy':
                    tolstoylm_toltest.append(languageModelTol.perplexity(test))
                elif author == 'wilde':
                    wildelm_toltest.append(languageModelWilde.perplexity(test))

        tolPercentage = calculateMinPerplexityPercentage(austenlm_toltest, dickenslm_toltest, tolstoylm_toltest, wildelm_toltest, 'tolstoy')

        #TESTING WILDE
        austenlm_wildetest = []
        dickenslm_wildetest = []
        tolstoylm_wildetest = []
        wildelm_wildetest = []

        tokenizedSentencesTestFile = []
        for sentence in developmentSetWilde:
            cleanSentence = preprocess(sentence)
            tokens = nltk.word_tokenize(cleanSentence)
            tokenizedSentencesTestFile.append(tokens)

        for author in authors:
            test_wilde, _ = padded_everygram_pipeline(n, tokenizedSentencesTestFile)
            for test in test_wilde:
                if author == 'austen':
                    austenlm_wildetest.append(languageModelAusten.perplexity(test))
                elif author == 'dickens':
                    dickenslm_wildetest.append(languageModelDickens.perplexity(test))
                elif author == 'tolstoy':
                    tolstoylm_wildetest.append(languageModelTol.perplexity(test))
                elif author == 'wilde':
                    wildelm_wildetest.append(languageModelWilde.perplexity(test))

        wildePercentage = calculateMinPerplexityPercentage(austenlm_wildetest, dickenslm_wildetest, tolstoylm_wildetest, wildelm_wildetest, 'wilde')

        print("Results on dev set:")
        print("austen      ", austenPercentage, "% correct")
        print("dickens      ", dickensPercentage, "% correct")
        print("tolstoy      ", tolPercentage, "% correct")
        print("wilde      ", wildePercentage, "% correct")
        ##########################################################################################################################################################


    else:
        austenlm_test = []
        dickenslm_test = []
        tolstoylm_test = []
        wildelm_test = []
        testFile =  processFile(sys.argv[3])
        cleanTextTest = re.sub(r"\n", '', testFile)
        sentencesTest = nltk.sent_tokenize(cleanTextTest)
        tokenizedSentencesTestFile = []
        for sentence in sentencesTest:
            cleanSentence = preprocess(sentence)
            tokens = nltk.word_tokenize(cleanSentence)
            tokenizedSentencesTestFile.append(tokens)
        for author in authors:
            test_data, _ = padded_everygram_pipeline(n, tokenizedSentencesTestFile)
            for test in test_data:
                if author == 'austen':
                    austenlm_test.append(languageModelAusten.perplexity(test))
                elif author == 'dickens':
                    dickenslm_test.append(languageModelDickens.perplexity(test))
                elif author == 'tolstoy':
                    tolstoylm_test.append(languageModelTol.perplexity(test))
                elif author == 'wilde':
                    wildelm_test.append(languageModelWilde.perplexity(test))
        calculateSentenceAuthor(austenlm_test, dickenslm_test, tolstoylm_test, wildelm_test)

    return


def main():
    if "-test" in sys.argv:
        extractDev = False
    else:
        extractDev = True
    fileNames = []
    with open(sys.argv[1], mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            fileNames.append(line.strip("\n"))
    buildLM(fileNames, extractDev)


if __name__ == '__main__':
   main()
