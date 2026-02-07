The zipped folder contains 3 python files - 1 each for the 3 different implementations of
the Needleman-Wunsch algorithm


1. To run NWA.py (standard Needleman-Wunsch with fixed scoring metric)
The command line for this file should be called in the following manner:

python NWA.py --seq1 Human_PAX.fa --seq2 Fly_PAX.fa

Note that the flags --seq1 and --seq2 must be provided while specifying the sequences. 
The ouput should print the alignment for sequence 1, the alignment for sequence 2, and the aligment score


2. To run NWA_B62.py (standard Needleman-Wunsch with BLOSUM62)
The command line for this file should be called in the following manner:

python NWA_B62.py --seq1 Human_PAX.fa --seq2 Fly_PAX.fa

Note that the flags --seq1 and --seq2 must be provided while specifying the sequences. 
The ouput should print the alignment for sequence 1, the alignment for sequence 2, and the aligment score


3. To run NWA_anchor.py (anchored Needleman-Wunsch with BLOSUM62)
The command line for this file should be called in the following manner:

python NWA_B62.py --seq1 Human_PAX.fa --seq2 Fly_PAX.fa --matches Match_PAX.txt

Note that the flags --seq1 and --seq2 must be provided while specifying the sequences. --matches is an optional parameter.
The ouput should print the alignment for sequence 1, the alignment for sequence 2, and the aligment score
