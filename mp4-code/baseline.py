"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    tags = {}
    wordTags = {}
    
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
            else:
                tags[t] += 1
    
    sentences = []
    mostOfenTags = max(tags, key=tags.get)

    for i in test:
        pairs = []
        for w in i:
            if w in wordTags:
                tag = max(wordTags[w], key=wordTags[w].get) #get the largest counts' tag of this word
                pairs.append((w, tag))
            else:
                pairs.append((w, mostOfenTags)) #unseen words with the tag that's seen the most often in training dataset.
        sentences.append(pairs)

    return sentences
    

