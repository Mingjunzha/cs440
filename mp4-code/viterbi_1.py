"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import numpy as np
import math
import sys
def viterbi_1(train, test):
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
    for i in wordTags.keys():
        for j in wordTags[i].keys():
            size = len(j)/10+1
            wordTags[i][j] = (wordTags[i][j]+emissionSmooth) / (tags[j] + emissionSmooth * size)

    initialProbabilities = np.zeros(index)
    transition = np.zeros((index, index))

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
        initialProbabilities[i] = (initialProbabilities[i]) / (len(train)/11.3+1)


    transitionSmooth= 0.00001 #sys.float_info.min # LaPlace smoothing
    tagSize = len(tags)/10+1
    for t, c in tags.items():
        prev = indexs[t]
        for i in range(len(transition)):
            transition[prev][i] = (transition[prev][i] + transitionSmooth) / (c + transitionSmooth * tagSize)

    tagList = []
    for tag in indexs.keys():
        tagList.append(tag)

    result = []

    for sentence in test:
        trellis = constructTrellis(sentence,wordTags,indexs,emissionSmooth,tags,tagList,initialProbabilities,transition)
        if len(trellis) == 0:
            result.append([])
            continue
        sentences = backtrackingTrellis(trellis,tagList,indexs)
        result.append(list(zip(sentence, sentences)))
    return result

def constructTrellis(sentence,wordTags,indexs,emissionSmooth,tags,tagList,initialProbabilities,transition):
    trellis = []
    tagSize = len(tags)/10+1
    for i in range(len(sentence)):
        newWord = sentence[i]
        pair = []
        if i == 0:
            if newWord in wordTags:
                for t in indexs.keys():
                    if t not in wordTags[newWord]:
                        prob = emissionSmooth / (tags[t] + emissionSmooth * tagSize)
                        tempTuple = (initialProbabilities[indexs[t]] * prob, t)
                        pair.append(tempTuple)
                    else:
                        prob = wordTags[newWord][t]
                        tempTuple = ((initialProbabilities[indexs[t]] * prob), t)
                        pair.append(tempTuple)
            else:
                for t in indexs.keys():
                    prob = emissionSmooth / (tags[t] + emissionSmooth * tagSize)
                    tempTuple = (initialProbabilities[indexs[t]] * prob, t)
                    pair.append(tempTuple)
        else:
            prob = 0
            for tag in indexs.keys():
                idx = indexs[tag]
                for j in range(len(indexs)):
                    prob = -sys.maxsize
                    if newWord in wordTags:
                        if tag in wordTags[newWord]:
                            prob = wordTags[newWord][tag]
                        else:
                            prob = emissionSmooth / (tags[tag] + emissionSmooth * tagSize)
                    else:
                        prob = emissionSmooth / (tags[tag] + emissionSmooth * tagSize)
                    probPrev = trellis[i - 1][idx][0]
                    prob = probPrev + math.log(transition[idx][j]) + math.log(prob)
                    tempTuple = (prob, tagList[idx])
                    if idx == 0:
                        pair.append(tempTuple)
                    elif (prob > pair[j][0]):
                        pair[j] = tempTuple
        trellis.append(pair)
    return trellis
def backtrackingTrellis(trellis,tagList,indexs):
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