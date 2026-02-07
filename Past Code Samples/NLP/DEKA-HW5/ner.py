#################### Instructions for running the program ####################
# In order to run this program, please call python ner.py
# By default, if a second argument is not provided, the program will test on the
# first 10 sentences of the WSJ corpus from NLTK. Otherwise, to test the first 'n'
# sentences of the corpus, please call python ner.py n, where n is an integer
#############################################################################


import re
import sys
import nltk
# nltk.download('averaged_perceptron_tagger')
from nltk import sent_tokenize, pos_tag, word_tokenize
from nltk.chunk import *
# nltk.download('treebank')
from nltk.corpus import treebank

def preprocess(input):
    print("Preprocessing text...")
    lower = input.lower()
    withoutPunct = re.sub(r'[^\w\s]', '', lower)
    return withoutPunct


def formatResult(inputTree):
    result = []
    for tag in inputTree:

        subTree = str(tag)
        if 'NP' in subTree:
            words = subTree.split(" ")
            for i in range(0, len(words)):
                if i == 0:
                    pass
                elif i == 1:
                    result.append((words[i].split("/")[0], "B-NP"))
                else:
                    result.append((words[i].split("/")[0], "I-NP"))

        else:
            result.append((tag[0], 'O'))

    print("RESULT IS ")
    print(result)
    return

def ner_bio_tag(input):
    word_tokens = word_tokenize(input)
    pos_tags = pos_tag(word_tokens)
    grammar =r"""NP: {<DT>?<JJ>*<NN>+}
                     {<DT>?<JJ>*<NNS>}
                     {<PRP\$>?<NN>}
                     {<PRP\$>?<NNS>}
                     {<NN><NN>+}
                     {<DT>?<NN>?<IN>?<DT>?<NN>?}"""
    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(pos_tags)
    formatResult(tree)
    return

def main():

    if len(sys.argv) > 1:
         numSentences = int(sys.argv[1])
    else:
        numSentences = 10

    wsj = nltk.corpus.treebank.sents()[:numSentences]

    for sentence in wsj:
        input = ' '.join(sentence)
        print("****************************************************")
        print("INPUT SENTENCE IS: ", input)
        preprocessedText = preprocess(input)
        ner_bio_tag(preprocessedText)
        print()




if __name__ == '__main__':
   main()
