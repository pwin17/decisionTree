## Pyone Thant Win
## AI and Machine Learning Lab C

import sys
import os.path
from math import log
from copy import deepcopy

def read_file(inputFile):
    """Converts inputfile into list of lines"""
    data = open(inputFile,"r")
    data = [line.split() for line in data.readlines()]
    return data[0], data[1:]

def get_frequency(data, resultAttr):
    """Returns +/- frequency of each attributes under all categories"""
    ## count yeses and nos
    frequencies = {}
    for each in data:
        if ((each[-1]) in frequencies):
            frequencies[each[-1]] += 1.0
        else:
            frequencies[each[-1]]  = 1.0
    return frequencies

def branch_frequency(categories, data, index):
    resultAttr = categories[-1]
    branchFreq = {}
    for each in data:
        # print(branchFreq)
        if each[index] in branchFreq:
            if each[-1] == 'yes':
                if 'yes' in branchFreq[each[index]]:
                    branchFreq[each[index]]['yes'] += 1
                else:
                    branchFreq[each[index]]['yes'] = 1
            else:
                if 'no' in branchFreq[each[index]]:
                    branchFreq[each[index]]['no'] += 1
                else:
                    branchFreq[each[index]]['no'] = 1
        else:
            branchFreq[each[index]] = {each[-1]: 1}
    return branchFreq

def entropy(frequencies, data):
    """Returns the entropy"""
    dataEntropy = 0.0
    total_count = len(data)
    freqVals = frequencies.values()
    for freq in frequencies.values():
        dataEntropy += (freq/total_count) * log(freq/total_count, 2)
    return dataEntropy*(-1)

def plurality_value(data,resultAttr):
	"""Return the predicted decision of example""" 
	major_freq = get_frequency(data,resultAttr)
	maximum = 0
	final = ""
	for key in major_freq.keys():
		if major_freq[key] > maximum:
			maximum = major_freq[key]
			final = key
	return final

def information_gain(data, parentEntropy, parentFreq, branchFreq):
    "Returns information gain of one category"
    remainder = 0.0
    parentTotal = parentFreq['yes'] + parentFreq['no']
    branchTotals = []
    for val in branchFreq.values():
        branchTotal = 0
        for count in val.values():
            branchTotal = branchTotal + count
        currentEntropy = entropy(val, data)
        remainder = remainder + (branchTotal/parentTotal * currentEntropy)
    infoGain = parentEntropy - remainder
    return infoGain

def best_category(categories, data, parentEntropy, parentFreq):
    "Returns the best category"
    bestInfoGain = 0.0
    bestCategory = ""
    for i in range(len(categories)-1):
        branchFreq = branch_frequency(categories, data, i)
        infoGain = information_gain(data, parentEntropy, parentFreq, branchFreq)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestCategory = categories[i]
    return bestCategory

def rank_categories(categories, data, parentEntropy, parentFreq):
    "Returns the ranked categories -- best to worst"
    ranks = []
    x = deepcopy(categories)
    while len(x) != 1:
        bestCategory = best_category(x, data, parentEntropy, parentFreq)
        ranks.append(bestCategory)
        x.remove(bestCategory)
    return ranks

def get_attributes(categories, data, bestAttr):
    """Returns all attributes under one category"""
    index_attr = categories.index(bestAttr)
    values = []
    for entry in data:
        if entry[index_attr] != bestAttr:
            if entry[index_attr] not in values:
                values.append(entry[index_attr])
    return values

def attr_dict(categories, data, ranking):
	"""Return category-attributes diictionary"""
	diction = {}
	new = deepcopy(categories)
	for item in ranking:
	 	values = get_attributes(new, data, item)
	 	diction[item] = values
	return diction

def find_by_attribute(categories, data, best, val):
    """Return data with matching attributes
        e.g 'tiny' -- [['white', 'pointed', 'yes', 'no'], ['white', 'pointed', 'yes', 'yes'], ['brown', 'pointed', 'yes', 'no']]"""
    similar = []
    index = categories.index(best)
    #index = 0
    for entry in data:
        #find entries with the give value
        if (entry[index] == val):
            newEntry = []
            #add value if it is not in best column
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            similar.append(newEntry)
    return similar

def learning(categories, data, resultAttr, parentData, originalCategories, originalData, parentFreq, parentEntropy):
    result = categories.index(resultAttr)
    resultValues = [row[result] for row in data] #list of final answers
    if not data: 
        x = plurality_value(parentData, resultAttr)
        return x
    elif len(categories) -1 <= 0: 
        x = plurality_value(data, resultAttr)
        return x
    elif resultValues.count(resultValues[0]) == len(resultValues): #all answers are same
        return resultValues[0]
    else:
        best = best_category(categories, data, parentEntropy, parentFreq)
        ranking = rank_categories(categories, data, parentEntropy, parentFreq)
        tree = {best: {}}
        attrs = attr_dict(originalCategories, originalData, ranking)
        for k, v in attrs.items():
            if k == best:
                for v2 in v:
                    similar = find_by_attribute(categories, data, best, v2)
                    new = deepcopy(categories)
                    new.remove(best)
                    subTree = learning(new, similar, resultAttr, data, originalCategories, originalData, parentFreq, parentEntropy)                  
                    tree[ranking[0]][v2] = subTree
    return tree

def print_tree(tree, resultAttr, tab=""):
    for key, value in tree.items():
        for k, v in value.items():
            print(tab, key, ": ", k)
            if type(v) == str:
                print(tab, tab, resultAttr, ": ", v)
            else:
                tabNew = tab + "\t"
                print_tree(v, resultAttr, tabNew)

def prediction(tree, test, categories):
    k = list(tree.keys())
    index = categories.index(k[0])
    for key, value in tree.items():
        for k, v in value.items():
            if k == test[index] and type(v) == str:
                return v
            elif k == test[index]:
                return prediction(v, test, categories)

def accuracy_training(tree, data, categories):
    correct = 0
    for row in data:
        predict = prediction(tree, row, categories)
        if predict == row[-1]:
            correct += 1
    percentage = correct / len(data) * 100
    return percentage

def accuracy_testing(categories, data):
    totalLines = len(data)
    totalCorrect = 0
    for i in range(totalLines):
        testLine = data[i]
        testData = data[:i] + data[i+1:]
        parentFreq = get_frequency(testData, categories[-1])
        parentEntropy = entropy(parentFreq, testData)
        tree = learning(categories, testData, categories[-1], None, categories, testData, parentFreq, parentEntropy)
        predict = prediction(tree, testLine, categories)
        if predict == testLine[-1]:
            totalCorrect += 1
    percentageAccuracy = totalCorrect/totalLines * 100
    return percentageAccuracy

def count_nodes(tree):
    c = 0
    for key, value in tree.items():
        c += 1
        if type(value) == str:
            c += 1
        else:
            c += count_nodes(value)
    return c

if __name__ == "__main__":
    if not (os.path.isfile(sys.argv[1])):
        print("error ", sys.argv[1], " is not valid")
        exit(-1)

    categories, data = read_file(sys.argv[1])
    parentFreq = get_frequency(data, categories[-1])
    parentEntropy = entropy(parentFreq, data)
    myTree = learning(categories, data, categories[-1], None, categories, data, parentFreq, parentEntropy)
    # print(myTree)
    print_tree(myTree, categories[-1])
    print("Total nodes: ",count_nodes(myTree))
    # print(prediction(myTree, data[9], categories))
    trainsetAccuracy = accuracy_training(myTree, data, categories)
    print("Training accuracy rate: ", round(trainsetAccuracy, 2), "%")
    accuracy = accuracy_testing(categories, data)
    print("Testing accuracy rate: ", round(accuracy, 2), "%")

    
    # print(myTree)
    # print(prediction(myTree, data[6], ranking))
    # print(parentEntropy)
    # # branchFreq = branch_frequency(categories, data, -3)
    # # print(branchFreq)
    # # print(information_gain(data, parentEntropy, parentFreq, branchFreq))
    # # best = best_category(categories, data, parentEntropy, parentFreq)
    # ranks = rank_categories(categories, data, parentEntropy, parentFreq)
    # # print(ranks)
    # value = attr_dict(categories, data, categories[-1], ranks)
    # # print(value)
    # # print(find_by_attribute(categories, data, ranks[0], 'tiny'))
    # branchFreq = group_by_index(data,0)
    # print(branchFreq)
    # allB = get_all_branchfreqs(categories, data)
    # print(allB)
    # allInfo = all_infoGain(categories, data, parentEntropy, parentFreq, allB)
    # print(allInfo)
    # best = rank_best(categories,allInfo)
    # print(best)
    # build_tree(best, allB)
    # print_tree(categories[-1], list(allB[best[0]]), best, allB)
    # tree(data, allB, best, categories)
    
    


'''This code assumes that the final result will always be in the final(rightmost) category.'''