import numpy as np
import math
import urllib.request
import pandas as pd

import argparse
from Bio import SeqIO

charList = []
scoreList = []
blosum62Dict = {}

currentLine = 0
with open('BLOSUM62.txt') as blosum:
    lines = blosum.readlines()

for line in lines:
    values = []
    if currentLine in (0,1, 2, 3, 4, 5):
        pass
    elif currentLine == 6:
        charList = line.split()
    else:
        values = line.split()[1:]
        values = list(map(int, values))
        scoreList.append(values)
    currentLine += 1

scoreList = np.array(scoreList)
rows = scoreList.shape[0]
columns = scoreList.shape[1]

for r in range(0, rows):
    for c in range(0, columns):
        blosum62Dict[(charList[r], charList[c])] = scoreList[r][c]



def computeMatchScore(letter1, letter2):
    if (letter1, letter2) == ('*','*'):
        return 1
    if '*' in (letter1, letter2):
        return (-4)
    if (letter1, letter2) in blosum62Dict.keys():
        return(blosum62Dict[(letter1, letter2)])
    else:
        return(blosum62Dict[(letter2, letter1)])




def getSequence(file):
    with open(file, "rt") as openedFile:
        for record in SeqIO.parse(openedFile, "fasta"):
            sequence = record.seq
    return(sequence)


def computeScoreMatrix(seq1, seq2):

    gapPen = -5

    ##Adding white space in front of sequence##
    editedSeq1 = " " + seq1
    editedSeq2 = " " + seq2

    ##Initializing the score matrix##
    m = len(editedSeq1)
    n = len(editedSeq2)
    scoreMat =[ [ 0 for i in range(m) ] for j in range(n) ]

    ##Initialize the first row and column as the base gap penalty, so that we##
    ##can use it for computing the rest of the matrix##
    for col in range(1, m):
        scoreMat[0][col] = col*gapPen
    for row in range(1, n):
        scoreMat[row][0] = row*gapPen

    ##Compute the scores for the rest of the matrix##
    for row in range(1, n):
        for col in range(1, m):
            matchPoints = computeMatchScore(editedSeq2[row], editedSeq1[col])
            scoreMat[row][col] = max(scoreMat[row-1][col-1] + matchPoints, scoreMat[row-1][col] + gapPen, scoreMat[row][col-1] + gapPen)


    return(scoreMat)


def computeAlignment(seq1, seq2):

    scoreMat = computeScoreMatrix(seq1, seq2)

    gapPen = -5


    seq1Alignment = ""
    seq2Alignment = ""

    ##Start from the last cell in the matrix##
    currentX = len(seq2)
    currentY = len(seq1)

    while (currentX > 0 or currentY > 0):

        if (seq2[currentX - 1] == seq1[currentY - 1] and currentX > 0 and currentY > 0) :
            seq1Alignment = seq1[currentY - 1] + seq1Alignment
            seq2Alignment = seq2[currentX - 1] + seq2Alignment
            currentX = currentX - 1
            currentY = currentY - 1

        if seq2[currentX - 1] != seq1[currentY - 1] :
            if (max(scoreMat[currentX-1][currentY-1], scoreMat[currentX-1][currentY], scoreMat[currentX][currentY-1]) == scoreMat[currentX-1][currentY-1] and currentX > 0 and currentY > 0) :
                seq1Alignment = seq1[currentY - 1] + seq1Alignment
                seq2Alignment = seq2[currentX - 1] + seq2Alignment
                currentX = currentX - 1
                currentY = currentY - 1
            elif (max(scoreMat[currentX-1][currentY-1], scoreMat[currentX-1][currentY], scoreMat[currentX][currentY-1]) == scoreMat[currentX-1][currentY] and currentX > 0):
                seq1Alignment = "-" + seq1Alignment
                seq2Alignment = seq2[currentX - 1] + seq2Alignment
                currentX = currentX - 1
            elif (max(scoreMat[currentX-1][currentY-1], scoreMat[currentX-1][currentY], scoreMat[currentX][currentY-1]) == scoreMat[currentX][currentY-1] and currentY > 0):
                seq1Alignment = seq1[currentY - 1] + seq1Alignment
                seq2Alignment = "-" + seq2Alignment
                currentY = currentY - 1
            else:
                pass
        else:
            pass

    print("Sequence 1 Alignment: ")
    print(seq1Alignment)
    print()
    print()
    print("Sequence 2 Alignment: ")
    print(seq2Alignment)
    print()
    print()
    print("Alignment Score: " ,scoreMat[len(seq2)][len(seq1)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq1')
    parser.add_argument('--seq2')
    args = parser.parse_args()

    seq1 = getSequence(args.seq1)
    seq2 = getSequence(args.seq2)

    computeAlignment(seq1, seq2)
