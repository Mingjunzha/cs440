# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
from nltk.stem import PorterStemmer
import string



def processWordList(reviews):
    result = []
    ps = PorterStemmer()
    for review in reviews:
        reviewResult = []
        for word in review:
            #processing each word, by strip, stemming and translate
            word = ps.stem(word.strip().translate(str.maketrans('', '', string.punctuation)))
            if word:
                reviewResult.append(word)
        if reviewResult:
            result.append(reviewResult)
    return result
def wordDict(train_set, train_labels):
    posDict = {}
    negDict = {}
    for i in range(len(train_set)):
        label = train_labels[i]
        review = train_set[i]
        for j in review:
            if label == 1:
                posDict[j] = posDict.get(j,0) + 1 # if no value, create value 0+1, otherwise, original value+1
            else:
                negDict[j] = negDict.get(j,0) + 1
    return posDict, negDict


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """

    train_set = processWordList(train_set)
    pos, neg = wordDict(train_set, train_labels)
  
    wordSize = len(set(i for sets in train_set for i in sets))/5+1

    dev_set = processWordList(dev_set)

    posSum = sum(pos.values())
    negSum = sum(neg.values())
    
    result = []
    for review in dev_set:
        posProb, negProb = 0,0
        for i in review:
            if (pos.get(i, 0)+ smoothing_parameter) != 0:
                posProb += np.log((pos.get(i, 0) + smoothing_parameter) / (posSum + smoothing_parameter * wordSize))
            if (neg.get(i, 0)+ smoothing_parameter) != 0:
                negProb += np.log((neg.get(i, 0) + smoothing_parameter) / (negSum + smoothing_parameter * wordSize))
        posProb += np.log(pos_prior)
        negProb += np.log(1-pos_prior)
        if posProb > negProb:
            result.append(1)
        else:
            result.append(0)

    return result

def reviewDict(train_set):
    revDict = {}
    for i in range(len(train_set)):
        review = set(train_set[i])
        for j in review:
            revDict[j] = revDict.get(j, 0) + 1
    return revDict

def reviewBigramDict(train_set):
    dicts = {}
    for i in range(len(train_set)):
        review = train_set[i]
        bigrams = [review[j:j + 2] for j in range(len(review) - 1)]
        newSet = set()
        for c in bigrams:
            if str(c) not in newSet: #check duplicate
                dicts[str(c)] = dicts.get(str(c), 0) + 1
                newSet.add(str(c))
    return dicts

def bigramDict(train_set, train_labels):
    pos = {}
    neg = {}
    for i in range(len(train_set)):
        label = train_labels[i]
        review = train_set[i]
        bigrams = [review[c:c + 2] for c in range(len(review) - 1)]
        for j in bigrams:
            if label == 1:
                pos[str(j)] = pos.get(str(j), 0) + 1
            else:
                neg[str(j)] = neg.get(str(j), 0) + 1
    return pos, neg


def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=0.02, bigram_lambda=0.02,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """

    train_set = processWordList(train_set)
    pos, neg = wordDict(train_set, train_labels)
    posSum = sum(pos.values())
    negSum = sum(neg.values())

    
 
    reviews = reviewDict(train_set)
    wordSize = len(set(i for sets in train_set for i in sets))/3.385+1

    bigramSize = 0
    for i in train_set:
        bigram = [i[j:j + 2] for j in range(len(i) - 1)]
        bigramSize += len(bigram)
    bigramSize = bigramSize+1
    dev_set = processWordList(dev_set)

    
    posBigram, negBigram = bigramDict(train_set, train_labels)
    posBiSum = sum(posBigram.values())
    negBiSum = sum(negBigram.values())
    

    reviewBigram =reviewBigramDict(train_set)
    tranSize = len(train_set)
    result = []
    for review in dev_set:
        posProb = 0
        negProb = 0
        for w in review:
            word = reviews.get(w, 0)
            num = 1
            if word != 0:
                num = np.log((tranSize + 1) / word)
            if (pos.get(w, 0) + unigram_smoothing_parameter) != 0:
                posProb += np.log((pos.get(w, 0) * num + unigram_smoothing_parameter) / (
                posSum + unigram_smoothing_parameter * wordSize))
            if (neg.get(w, 0) + unigram_smoothing_parameter) != 0:
                negProb += np.log((neg.get(w, 0) * num + unigram_smoothing_parameter) / (
                negSum + unigram_smoothing_parameter * wordSize))

        posProb += np.log(pos_prior)
        negProb += np.log(1-pos_prior)
        #bigrams
        posBiProb = 0
        negBiProb = 0

        bigReview = [review[c:c + 2] for c in range(len(review) - 1)]
        for bigr in bigReview:
            key = str(bigr)
            word = reviewBigram.get(key, 0)
            num = 1
            if word != 0:
                num = np.log((tranSize + 1) / word)
            if (posBigram.get(key, 0) + bigram_smoothing_parameter) != 0:
                posBiProb += np.log((posBigram.get(key, 0) * num + bigram_smoothing_parameter) / (
                    posBiSum + bigram_smoothing_parameter * bigramSize))
            if (negBigram.get(key, 0) +bigram_smoothing_parameter) != 0:
                negBiProb += np.log((negBigram.get(key, 0) * num + bigram_smoothing_parameter) / (
                    negBiSum + bigram_smoothing_parameter * bigramSize))
        posBiProb += np.log(pos_prior)
        negBiProb += np.log(1-pos_prior)
        pProb = (1-bigram_lambda) * posProb + posBiProb * bigram_lambda
        nProb = (1-bigram_lambda) * negProb + negBiProb * bigram_lambda

        if pProb > nProb:
            result.append(1)
        else:
            result.append(0)
    return result

