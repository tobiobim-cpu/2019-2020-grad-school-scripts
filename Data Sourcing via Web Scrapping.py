###Check the Jupyter files


##BeautifulSoup object -> rep of the doc you are scrapping as a whole, easily navigatable and searchable
###BeautifulSoup uses NavigatableString class as a container for bits of text
##Tag Objects -> XML and HTML elements in an original doc
##NavigatableString(NS) object -> adds a bit of text within a tag
##Comment object -> a type of NS object that you can use for commenting your code

import sys
print(sys.version)

from bs4 import BeautifulSoup


## Navigatable stringobjects are used as containers to store texts that are stored inside of tags


#####Introduction to Natural language Processing (NLP)
##Sentence Tokenization: Is the process of breaking down paragraphs or a complete set of text into sentences
##A plain sentence would be tokenized into an array
##Word tokenization: Is breaking down a paragraph, a sentence, or a complete text corpus into array of words
#Stop words: add little to no meaning to a body of a text at large (They are noise in the body of a text)
##We use NLTK library
##Stemming: process for reducing the size of the corpus by convering words to their root word(basically linguistic normalization)
##Root word: a stem word that may or may not have meaning when taken alone(derived from other words-Running -> run[root word])
##Base word: a stem word that has meaning even when alone(only prefixes and suffixes being attached)
##Lemmatizing: is the process of reducing a word to its base word (lemmatizing 'better' -> 'good')
##Part of Speech tagging: process of identifying a part of speech within a given body of text
##Frequency Distr plotting: for quantifying  textual corpus. Counts and plot frequency of words in a given text
####It can be used for sentiment analysis and to understand the distributions of words



