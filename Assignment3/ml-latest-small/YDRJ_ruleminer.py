import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('ratings.csv')
df=df.loc[df['rating']>2]
df.loc[df['userId']==442]
df.drop([68317,68336], axis=0, inplace=True)
df.loc[df['userId']==508]
df.drop([80352,80353,80357,80358,80363,80364], axis=0, inplace=True)
df_1=df.loc[df['userId']==1]
train1, test1 = train_test_split(df_1, test_size=0.2)
df_train,df_test=train_test_split(df,test_size=0.2,stratify=df['userId'])
df.drop(columns=['rating','timestamp'],inplace=True)
train_dataset = df_train.groupby(by = ['userId'])['movieId'].apply(list).reset_index()
train_dataset = train_dataset['movieId'].tolist()
test_dataset = df_test.groupby(by = ['userId'])['movieId'].apply(list).reset_index()
test_dataset = test_dataset['movieId'].tolist()
merge_list2 = df_test.groupby(by = ['userId'])['movieId'].apply(list).reset_index()
merge_list2 = merge_list2['movieId'].tolist()

# coding apriori algorithm
def apriori(train_dataset, minSupport = 0.05):
    C1 = createC1(train_dataset)
    D = list(map(frozenset, train_dataset))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def createC1(train_dataset):
    C1 = []
    for transaction in train_dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2:
                retList.append(Lk[i] | Lk[j])
    return retList

# coding association rules
def generateRules(L, supportData, minConf=0.1):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.1):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'confidence:',conf,'support:',supportData[freqSet])
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.1):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

L, suppData = apriori(train_dataset)
rules = generateRules(L, suppData, minConf=0.1)

# select top 100 rules sorted by confidence
rules.sort(key=lambda x: x[2], reverse=True)
rules[:100]
print()
print()
print()
print()
print()
print()
print("Top 100 rules sorted by confidence:")
for i in range(100):
    print(rules[i])


with open(r'confidence.txt', 'w') as fp:
    fp.write("\n".join(str(item) for item in rules))

# select top 100 rules sorted by support
rules.sort(key=lambda x: suppData[x[0]|x[1]], reverse=True)
rules[:100]
print()
print()
print()
print()
print()
print()
print("Top 100 rules sorted by support:")
for i in range(100):
    print(rules[i])
with open(r'support.txt', 'w') as fp:
    fp.write("\n".join(str(item) for item in rules))

# select rules which are common in both top 100 rules sorted by confidence and support
rules1 = rules[:100]
rules2 = rules[:100]
rules1.sort(key=lambda x: x[2], reverse=True)
rules2.sort(key=lambda x: suppData[x[0]|x[1]], reverse=True)
#get common rules from rules1 and rules2
common_rules = []
for i in range(100):
    for j in range(100):
        if rules1[i][0] == rules2[j][0] and rules1[i][1] == rules2[j][1]:
            common_rules.append(rules1[i])
            break
print()
print()
print()
print()
print()
print()
print("Common rules in top 100 rules sorted by confidence and support:")
for i in range(len(common_rules)):
    print(common_rules[i])

#export common rules in a text file
with open(r'common.txt', 'w') as fp:
    fp.write("\n".join(str(item) for item in common_rules))

#  For each user in the test set, select association rules of the form X→Y,
# where X is the movie in the training set.
def predict(test_dataset, rules):
    predicted = []
    for i in range(len(test_dataset)):
        for j in range(len(rules)):
            if list(rules[j][0])[0] in test_dataset[i]:
                predicted.append(list(rules[j][1])[0])
                break
    return predicted

# Compute the average precision and average recall by varying the number of rules from 1 to 10 and plot the graphs.
nom=[1,2,3,4,5,6,7,8,9,10]
def compute_precision_recall(test_dataset, rules):
    precision = []
    recall = []
    for i in range(1,11):
        predicted = predict(test_dataset, rules[:i])
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(test_dataset)):
            for j in range(len(predicted)):
                if predicted[j] in test_dataset[i]:
                    TP += 1
                else:
                    FP += 1
            FN = abs(len(test_dataset[i]) - TP)
        precision.append(TP/(TP+FP))
        recall.append(TP/(TP+FN))
        print("Recall:", recall,TP,FN)

    return precision, recall
from matplotlib import pyplot as plt
precision, recall = compute_precision_recall(test_dataset, rules)
plt.plot(precision)
plt.plot(recall)
plt.legend(['precision', 'recall'])
plt.show()

#select a random user and calculate the precision and recall for the user
import random
user = random.randint(0, len(test_dataset))
print("User:", user)
print("Movies watched by user:", test_dataset[user])
predicted = predict(test_dataset, rules)
print("Predicted movies for user:", predicted)
TP = 0
FP = 0
FN = 0
for i in range(len(test_dataset[user])):
    if predicted[i] in test_dataset[user]:
        TP += 1
    else:
        FP += 1
FN = len(test_dataset[user]) - TP
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("Precision:", TP/(TP+FP))
print("Recall:", TP/(TP+FN))

