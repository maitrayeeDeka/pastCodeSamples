import re
import sys


#This homework was worked on by Monika Bhardwaj and Maitrayee Deka


def word_tokenize(input):

    cleanInput = re.sub(r",|!|\?", '', input)  #removing commas, exclamation points and question marks so they are not part of the words
    cleanInput =  re.sub(r"\n", ' ', cleanInput) #replacing \n with space (so that splitting still works and words aren't concatenated)
    whitespace = r"\S+"
    output = re.findall(whitespace, cleanInput)

    print(output)

    return output


def sent_tokenize(input):
    cleanInput =  re.sub(r"\n", ' ', input)
    pattern = r"\?|\.|!"
    output = re.split(pattern, cleanInput)

    print(output)

    return output



def main():
    with open(sys.argv[1], encoding="utf8") as f:
        lines = f.readlines()
        inputString = ''.join(lines)

        print("********************************************************************************")
        print()
        print("Word tokenization shown below:")
        print()
        word_tokenize(inputString)
        print()
        print("********************************************************************************")

        print()
        print()
        
        print("********************************************************************************")
        print()
        print("Sentence tokenization shown below:")
        print()
        sent_tokenize(inputString)
        print()
        print("********************************************************************************")


if __name__ == '__main__':
   main()
