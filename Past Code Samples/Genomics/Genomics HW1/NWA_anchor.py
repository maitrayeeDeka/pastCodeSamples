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


def getUnalignedSequences(file, seqNum, matches):
    seq = getSequence(file)
    unalignedSequences = {}
    matchMat = np.loadtxt(matches, dtype=int, usecols=range(4))
    rows = matchMat.shape[0]
    matchLocs = [0]
    for i in range(0, rows):
        if seqNum == 1:
            matchLocs.append(matchMat[i][0] - 1)
            matchLocs.append(matchMat[i][1] + 1)

        else:
            matchLocs.append(matchMat[i][2] - 1)
            matchLocs.append(matchMat[i][3] + 1)
    matchLocs.append(len(seq))

    while len(matchLocs) != 0:
        unalignedSequences[matchLocs[0]] = seq[matchLocs[0]: matchLocs[1] + 1]
        matchLocs.pop(0)
        matchLocs.pop(0)

    return(unalignedSequences)


def getAlignedSequences(file, seqNum, matches):
    seq = getSequence(file)
    alignedSequences = {}
    matchMat = np.loadtxt(matches, dtype=int, usecols=range(4))
    rows = matchMat.shape[0]
    matchLocs = []
    for i in range(0, rows):
        if seqNum == 1:
            matchLocs.append(matchMat[i][0])
            matchLocs.append(matchMat[i][1])

        else:
            matchLocs.append(matchMat[i][2])
            matchLocs.append(matchMat[i][3])


    while len(matchLocs) != 0:
        alignedSequences[matchLocs[0]] = seq[matchLocs[0]: matchLocs[1] + 1]
        matchLocs.pop(0)
        matchLocs.pop(0)

    return(alignedSequences)



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

    gapPen = -2
    matchScore = 1
    mismatchPen = -3

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

    return(seq1Alignment, seq2Alignment, scoreMat[len(seq2)][len(seq1)])
    # print(seq1Alignment, seq2Alignment)
    # print(scoreMat[len(seq2)][len(seq1)])


def computeAnchoredScore(unalignedSeq1, unalignedSeq2, alignedSeq1, alignedSeq2):
    unalignedScore = 0
    alignedScore = 0

    for i in range(0, len(unalignedSeq1)):
        tempScore = 0
        tempScore = computeScoreMatrix(unalignedSeq1[i], unalignedSeq2[i])[len(unalignedSeq2[i])][len(unalignedSeq1[i])]
        unalignedScore = unalignedScore + tempScore

    for i in range(0, len(alignedSeq1)):
        tempScore = 0
        tempScore = computeScoreMatrix(alignedSeq1[i], alignedSeq2[i])[len(alignedSeq2[i])][len(alignedSeq1[i])]
        alignedScore = alignedScore + tempScore

    return (unalignedScore + alignedScore)


def computeAnchoredAlignment(seq1, unalignedSeq1, alignedSeq1, seq2, unalignedSeq2, alignedSeq2):
    allSeq1 = unalignedSeq1
    allSeq1.update(alignedSeq1)
    sortedSeq1Keys = sorted(allSeq1)
    sequence1Chunks = []

    for i in range(0, len(sortedSeq1Keys)):
        sequence1Chunks.append(allSeq1[sortedSeq1Keys[i]])


    allSeq2 = unalignedSeq2
    allSeq2.update(alignedSeq2)
    sortedSeq2Keys = sorted(allSeq2)
    sequence2Chunks = []

    for i in range(0, len(sortedSeq2Keys)):
        sequence2Chunks.append(allSeq2[sortedSeq2Keys[i]])

    alignmentSeq1 = ""
    alignmentSeq2 = ""
    alignmentScore = 0


    for i in range(0, len(sequence1Chunks)):
        result = computeAlignment(sequence1Chunks[i], sequence2Chunks[i])

        alignmentSeq1 = alignmentSeq1 + result[0]
        alignmentSeq2 = alignmentSeq2 + result[1]
        alignmentScore = result[2] + alignmentScore

        print(alignmentScore)

    # print("Sequence 1 Alignment: ")
    # print(alignmentSeq1)
    # print()
    # print()
    # print("Sequence 2 Alignment: ")
    # print(alignmentSeq2)
    # print()
    # print()
    # print("Alignment Score: ", alignmentScore)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq1')
    parser.add_argument('--seq2')
    parser.add_argument('--matches', required=False)
    args = parser.parse_args()



    if args.matches is not None:

        seq1 = getSequence(args.seq1)
        seq2 = getSequence(args.seq2)
        unalignedSeq1 = getUnalignedSequences(args.seq1, 1, args.matches)
        unalignedSeq2 = getUnalignedSequences(args.seq2, 2, args.matches)
        alignedSeq1 = getAlignedSequences(args.seq1, 1, args.matches)
        alignedSeq2 = getAlignedSequences(args.seq2, 2, args.matches)

        computeAnchoredAlignment(seq1, unalignedSeq1, alignedSeq1, seq2, unalignedSeq2, alignedSeq2)




    else:
        seq1 = getSequence(args.seq1)
        seq2 = getSequence(args.seq2)
        result = computeAlignment(seq1, seq2)

        print("Sequence 1 Alignment: ", result[0])
        print()
        print()
        print("Sequence 2 Alignment: ", result[1])
        print()
        print()
        print("Alignment Score: ", result[2])
