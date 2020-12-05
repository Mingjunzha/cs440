"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import numpy as np
import sys
import math
def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    wordTags = {}
    tags = {}

    index = 0
    indexs = {}
    
    for sentence in train:
        for data in sentence:
            w, t = data
            if w not in wordTags:
                wordTags[w] = {}
            if t not in wordTags[w]:
                wordTags[w][t] = 1
            else:
                wordTags[w][t] += 1
            
            if t not in tags:
                tags[t] = 1
                indexs[t] = index
                index += 1
            else:
                tags[t] += 1
    emissionSmooth = 0.00001
    hapax = {}
    for i in wordTags.keys():
        if len(wordTags[i].keys()) == 1:
            for j in wordTags[i].keys():
                hapax[j] = hapax.get(j,0)+1
    print(hapax)
    open = ["NOUN", "VERB","ADJ","ADV"]
    for i in wordTags.keys():
        for j in wordTags[i].keys():
            if j in open:
                emissionSmooth = 0.00001
            else:
                emissionSmooth = 0.135
            wordTags[i][j] = (wordTags[i][j]+emissionSmooth) / (tags[j] + emissionSmooth * (len(j)+1))
  

    initialProbabilities = np.zeros(index)
    transition = np.zeros((index, index))
    mostOfenTags = max(tags, key=tags.get)
    for i in train:
        flag = True
        for j in range(len(i[:-1])):
            w, t = i[j]
            curr = indexs[t]
            if flag:
                initialProbabilities[curr] += 1
                flag = False
            next = i[j + 1][1]
            transition[curr][indexs[next]] += 1


    # get initial probabilities
    for i in range(len(initialProbabilities)):
        initialProbabilities[i] = (initialProbabilities[i]) / (len(train)+ 1)

    tagSize = len(tags)+1
    transitionSmooth= 0.00001  # LaPlace smoothing
    for t, c in tags.items():
        j = indexs[t]
        for i in range(len(transition)):
            transition[j][i] = (transition[j][i] + transitionSmooth) / (c + transitionSmooth * tagSize)

    tagList = []
    for tag in indexs.keys():
        tagList.append(tag)

    result = []
    for sentence in test:
        trellis = constructTrellis(sentence,wordTags,indexs,emissionSmooth,tags,tagList,initialProbabilities,transition,mostOfenTags,hapax)
        if len(trellis) == 0:
            result.append([])
            continue
        
        sentences = backtracingTrellis(trellis,tagList,indexs)
        result.append(list(zip(sentence, sentences)))
    return result

def checkTags(listOfTags,word,tag,tags,emissionSmooth,tagSize,mostOfenTags,hapax):
    hapaxs = sum(hapax.values())+1
    if tag in hapax:
        prob = (hapax.get(tag)+emissionSmooth) / (hapaxs+emissionSmooth * tagSize)
    else:
        prob = (emissionSmooth) / (hapaxs+emissionSmooth * tagSize)
    return prob
def checkSmooth(t,open):
    if t in open:
        emissionSmooth = 0.00001
    else:
        emissionSmooth = 0.135
    return emissionSmooth
def constructTrellis(sentence,wordTags,indexs,emissionSmooth,tags,tagList,initialProbabilities,transition,mostOfenTags,hapax):
    trellis = []
    tagSize = len(tags) + 1
    listOfTags = {"ly":["ADV"],"ing":["NOUN","VERB"],"able":["ADJ","NOUN"],"ment":["NOUN"],"ed":["VERB","NOUN"],"on":["NOUN","IN","ADV"],"s":["NOUN","VERB","ADV","IN","CONJ"]}
    open = ["NOUN", "VERB","ADJ","ADV"]
    for i in range(len(sentence)):
        newWord = sentence[i]
        pair = []
        if i == 0:
            if newWord in wordTags:
                for t in indexs.keys():
                    emissionSmooth = checkSmooth(t,open)
                    if t not in wordTags[newWord]:
                        prob = (emissionSmooth) / (tags[mostOfenTags]+emissionSmooth * tagSize)
                        tempTuple = (prob, t)
                        pair.append(tempTuple)
                    else:
                        prob = wordTags[newWord][t]
                        tempTuple = (initialProbabilities[indexs[t]] * prob, t)
                        pair.append(tempTuple)
            else:
                for t in indexs.keys():
                    emissionSmooth = checkSmooth(t,open)
                    prob = checkTags(listOfTags,newWord,t,tags,emissionSmooth,tagSize,mostOfenTags,hapax)
                    tempTuple = (prob, t)
                    pair.append(tempTuple)
        else:
            prob = 0
            for tag in indexs.keys():
                idx = indexs[tag]
                emissionSmooth = checkSmooth(tag,open)
                for j in range(len(indexs)):
                    prob = -999999
                    if newWord in wordTags:
                        if tag in wordTags[newWord]:
                            prob = wordTags[newWord][tag]
                        else:
                            prob = prob = (emissionSmooth) / (tags[mostOfenTags]+emissionSmooth * tagSize)
                            
                    else:
                        prob = checkTags(listOfTags,newWord,tag,tags,emissionSmooth,tagSize,mostOfenTags,hapax)
                    #if (i == 1):
                    prob = trellis[i - 1][idx][0]+ math.log(transition[idx][j]) + math.log(prob)
                    #else:
                    #prob = trellis[i - 1][idx][0] + trellis[i - 2][idx][0] + math.log(transition[idx][j]) + math.log(prob)
                    tempTuple = (prob, tagList[idx])
                    if idx == 0:
                        pair.append(tempTuple)
                    elif (prob > pair[j][0]):
                        pair[j] = tempTuple
        trellis.append(pair)
    return trellis
def backtracingTrellis(trellis,tagList,indexs):
    sentences = []
    
    tupList = trellis[len(trellis) - 1]
    i = tupList.index((max(tupList)))
    sentences.append(tagList[i])
    tagPrev = max(tupList)
    for i in range(len(trellis)-1, 0, -1): #backtracing
        tagPrev = trellis[i - 1][indexs[tagPrev[1]]]
        sentences.insert(0, tagPrev[1])
    maxTag = max(trellis[0])[1]
    sentences[0] = maxTag
    return sentences