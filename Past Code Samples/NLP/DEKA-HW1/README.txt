My method for extracting topic words from the text involves the 'NTLK' library.

If this library isn't already installed, it can be installed via the command 'pip install nltk' (for python).

For my method, I first use the list of 'NLTK' stopwords to remove any stop word within the text file, and then
strip the text of punctuations as well. Finally, I use the 'Collections' library to count the most frequent
words in the text. I then print the top 5 most common words.


