#Madan Bhurtel
#1001752499

import nltk as NLTK
from nltk.tokenize import RegexpTokenizer as TOKENIZER
from nltk.corpus import stopwords as STOPWORDS
from nltk.stem.porter import PorterStemmer as PS
import os as OS
import math as MATH 
import time

t1= time.time()
# Tokenizer for regex calc, only gets words, numebrs are neglected
presidential_token = TOKENIZER(r'[a-zA-Z]+')

# Porter Stemmer
stemmer = PS()

# Store all the documents
record = []

# Index of filename to the record array
# eg: "abc.txt": 0
filename_index = {}
# Array where filename is stored at the given indices, translated given index to a filename
# Opposite of what filename_index does eg: at index 0 --> "abc.txt"
index_filename = []

# Convert all the stop words into a hashmap for O(1) lookups
stopwords = {}
for word in STOPWORDS.words('english'):
    stopwords[word] = 1

# Get the idf of a token
def getidf(token):
    df = 0          #Number of documents with term t 
    N = 0           #Total number of documents
    for document in record:
        N += 1
        if token in document:
            df += 1
    return MATH.log10(N/df) if df != 0 else -1        #return idf of token if document frequency is not 0, otherwise, -1 

# Get the weight for a given term in a specified document
def getweight(filename, token):
    return (record[filename_index[filename]]).get(token, 0)

# Calculates the tf_idf score for all the documents specified a path.
# Main function with core logic is here
def calculate_tf_idf(path):
    NUMBER_OF_DOCUMENTS = 0

    # Loop through all the files given to a path
    for files in OS.listdir(path):
        current_file = open(OS.path.join(path, files), "r", encoding='UTF-8')
        document = current_file.read()
        current_file.close()
        document = document.lower()
        # Calculate the term frequency for all the terms in the document and append to the list
        record.append(calculate_term_frequency(document))
        # Maintain maps to go from filename to index and back
        filename_index[files] = NUMBER_OF_DOCUMENTS
        index_filename.append(files)
        NUMBER_OF_DOCUMENTS += 1

    # Now calculate the tf_idf weight of each terms
    for doc in record:
        # Normalize the term
        square_total = 0
        for terms in doc:
            # Initialize the df to zero and loop through to search for no of occurence of terms
            df = 0

            # Current term frequency in the document is already calculated before
            tf = doc[terms]

            # calculate the number of documents that this word appears in
            for document_traversal in record:
                if terms in document_traversal:
                    df += 1

            # calculate the tf-idf here
            tf_idf = (
                1 + MATH.log10(tf)) * MATH.log10(NUMBER_OF_DOCUMENTS/df)
            doc[terms] = tf_idf
            square_total += tf_idf ** 2

        # Normalize all of the terms
        for terms in doc:
            doc[terms] = doc[terms] / MATH.sqrt(square_total)

    return record


def calculate_term_frequency(document):
    # Remove stop words in O(N) where N is the length of the document
    # Also perform stemming immediately

    # Calculate the terms frequency in the document
    term_vector = {}
    for word in presidential_token.tokenize(document):
        if word not in stopwords:
            stemmed = stemmer.stem(word)
            if stemmed in term_vector:
                term_vector[stemmed] += 1
            else:
                term_vector[stemmed] = 1

    return term_vector


# Function that returns a tupple of document and cosine similarity of a query string
def query(qstring):

    # Helper Function
    # Get the score of the given document in a posting list,
    # if not in the list return 10th elem weight
    # And T/F denoting if the score is the element's actual score
    def posting_list_get_score(document, posting_list):
        for elements in posting_list:
            # If element is found in the posting list return its score, and T for True score
            if elements[0] == document:
                return elements[1], True
        # If element is not found, return the last element's score and false for truth score
        return posting_list[len(posting_list)-1][1], False

    # Lowercase, Tokenize and Stem the query
    qstring = [stemmer.stem(token)
               for token in presidential_token.tokenize(qstring.lower())]
    query_word = {}

    # Calculate the term frequency for the query string
    for word in qstring:
        if word in query_word:
            query_word[word] += 1
        else:
            query_word[word] = 1

    # Normalize the query word
    sum_of_square = 0
    for terms in query_word:
        query_word[terms] = 1 + MATH.log10(query_word[terms])
        sum_of_square += query_word[terms] ** 2
    sum_of_square = MATH.sqrt(sum_of_square)

    query_vector = {
        count: (query_word[count])/sum_of_square for count in query_word}

    # in the form of document index, score
    # One liner to get the posting list ommitted as graded based on modularity of code
    # posting_list = [(sorted([[ doc, (record[doc]).get(terms, 0)] for doc in range(len(record))],
    #                        reverse=True, key=lambda x: x[1]))[0:10] for terms in query_vector]

    posting_list = []

    # Return the top-10 elements in corresponding posting list for each term of the query word
    for terms in query_word:
        term_posting_list = []
        # Get all the score for every documents
        for documents_index in range(len(record)):
            documents = record[documents_index]
            # Append a (filename, score) tupple to the posting list
            term_posting_list.append((index_filename[documents_index], getweight(
                index_filename[documents_index], terms)))
        # Sort the list and only put 10 items into it
        term_posting_list = sorted(
            term_posting_list, reverse=True, key=lambda x: x[1])[0:10]
        posting_list.append(term_posting_list)
        #print(posting_list)

    # Store all the documents that occur in any of the posting list
    documents_in_posting_list = {
        doc[0]: 0 for elem in posting_list for doc in elem}

    # Calculate the similarity score of all the documents
    sim = []
    for a_document in documents_in_posting_list:
        similarity = 0
        acutual_score = True
        for term_score, single_posting_list in zip(query_vector, posting_list):      #zip(iterator1, iterator 2)
            cosine_score, true_score = posting_list_get_score(
                a_document, single_posting_list)
            similarity += cosine_score * query_vector[term_score]
            if not true_score:
                acutual_score = False
        sim.append((a_document, similarity, acutual_score))
        #print(sim)

    # O(N) where N is the total length of numbers of documents in the posting list
    maximum_cosine_similarity_document = None

    # Find the document with maximum cosine similiarity
    for score in sim:
        if maximum_cosine_similarity_document:
            if maximum_cosine_similarity_document[1] < score[1]:
                maximum_cosine_similarity_document = score
        else:
            maximum_cosine_similarity_document = score

    if maximum_cosine_similarity_document[2]:
        if maximum_cosine_similarity_document[1] == 0.0:
            return (None, maximum_cosine_similarity_document[1])
        return (maximum_cosine_similarity_document[0], maximum_cosine_similarity_document[1])
    else:
        return("fetch more", 0)

calculate_tf_idf("./presidential_debates")
t2= time.time()

if __name__ == "__main__":
    print('\n           Idf:        ')
    print("Idf health: %.12f" % getidf("health"))
    print("Idf agenda: %.12f" % getidf("agenda"))
    print("Idf vector: %.12f" % getidf("vector"))
    print("Idf reason: %.12f" % getidf("reason"))
    print("Idf hispan: %.12f" % getidf("hispan"))
    print("Idf hispanic: %.12f" % getidf("hispanic"))
    '''
    #REST TEST CASE 
    print("%.12f" % getidf('andropov'))
    #1.477121254720
    print("%.12f" % getidf('identifi'))
    #0.330993219041
    '''

    print('\n         Normalize td-idf(weight):             ')
    print("Weight '2012-10-03.txt' 'health': %.12f" % getweight("2012-10-03.txt", "health"))
    print("Weight '1960-10-21.txt' 'reason': %.12f" % getweight("1960-10-21.txt", "reason"))
    print("Weight '1976-10-22.txt' 'agenda': %.12f" % getweight("1976-10-22.txt", "agenda"))
    print("Weight '2012-10-16.txt' 'hispan': %.12f" % getweight("2012-10-16.txt", "hispan"))
    print("Weight '2012-10-16.txt' 'hispanic': %.12f" % getweight("2012-10-16.txt", "hispanic"))
    '''
    #REST TEST CASE 
    print("%.12f" % getweight('1960-09-26.txt','accomplish'))
    #0.012765709568
    print("%.12f" % getweight('1960-10-07.txt','andropov'))
    #0.000000000000
    print("%.12f" % getweight('1960-10-13.txt','identifi'))
    #0.013381177224
    '''
    print('\nQuery:')
    print("(%s, %.12f)" % query("health insurance wall street"))
    print("(%s, %.12f)" % query("particular constitutional amendment"))
    print("(%s, %.12f)" % query("terror attack"))
    print("(%s, %.12f)" % query("vector entropy"))
    print('\n')
    caltime = t2-t1
    print("Totaltime taken: %.5f" %caltime)