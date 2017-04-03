import numpy as np
import csv
from sklearn import svm

# Connection between users
userLink = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                     [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                     [0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                     [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
                     [1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                     [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                     [0, 0, 1, 0, 1, 0, 0, 1, 1, 0]])

# Gender for each users
userGender = []
with open("testNetwork.txt") as csvFile:
    readCSV = csv.reader(csvFile, delimiter="\t")
    for row in readCSV:
        userGender.append(int(row[3]))

# Random 50% to be private
a = [1] * 5
b = [0] * 5
userStatus = np.concatenate((a, b))
np.random.shuffle(userStatus)

# Attack without links (BASIC)
print("#### BASIC ####")
accuracy = []
for i in range(0,len(userGender)):
    # Do for only private user
    if userStatus[i] == 0:
        Target = i
        countMale = 0
        countFemale = 0
        countPublic = 0
        for idx, val in enumerate(userGender):
            if idx != Target and userStatus[idx] != 0:
                if userGender[idx] == 0:
                    countMale += 1
                else:
                    countFemale += 1
                countPublic += 1
        # print("Pop Male: {0}, Pop Female: {1}".format(float(countMale)/countPublic, float(countFemale)/countPublic))
        predictGender = 0 if countMale > countFemale else 1
        # print ("Prediction Output: {0}\n".format(predictGender == userGender[Target]))
        accuracy.append(predictGender == userGender[Target])
print("Accuracy: {0}".format(accuracy.count(True)/float(len(accuracy))))

# Attack with Friend-aggregate model (AGG)
print("\n#### AGG ####")
accuracy = []
userGenderPredict = []
for i in range(0,len(userGender)):
    # Do for only private user
    if userStatus[i] == 0:
        Target = i
        countMale = 0
        countFemale = 0
        countPublic = 0
        for idx, val in enumerate(userGender):
            # Consider only public user
            if idx != Target and userStatus[idx] != 0 and userLink[idx,Target] == 1:
                if userGender[idx] == 0:
                    countMale += 1
                else:
                    countFemale += 1
                countPublic += 1
        #print("Pop Male: {0}, Pop Female: {1}".format(float(countMale)/countPublic, float(countFemale)/countPublic))
        predictGender = 0 if countMale > countFemale else 1
        userGenderPredict.append(predictGender)
        #print ("Prediction Output: {0}".format(predictGender is userGender[Target]))
        accuracy.append(predictGender == userGender[Target])
    else:
        userGenderPredict.append(userGender[i])
print("Accuracy: {0}".format(accuracy.count(True)/float(len(accuracy))))

# Attack with Collective classification model (CC)
print("\n#### CC ####")
accuracy = []
# Train collective classifier
for j in range(0, 10):
    userGenderPredictNew = []
    for i in range(0,len(userGender)):
        # Do for only private user
        if userStatus[i] == 0:
            Target = i
            countMale = 0
            countFemale = 0
            countPublic = 0
            for idx, val in enumerate(userGender):
                # Consider both private and public
                if idx != Target and userLink[idx, Target] == 1:
                    if userGenderPredict[idx] == 0:
                        countMale += 1
                    else:
                        countFemale += 1
                    countPublic += 1
            predictGender = 0 if countMale > countFemale else 1
            userGenderPredictNew.append(predictGender)
        else:
            userGenderPredictNew.append(userGender[i])
    userGenderPredict = userGenderPredictNew
for i in range(0, 10):
    if userStatus[i] == 0:
        accuracy.append(userGenderPredict[i] == userGender[i])
print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))

#Attack with Flat-link model (LINK) with SVM
print("\n#### LINK ####")
trainX = []
trainY = []
testX = []
testY = []
for idx, data in enumerate(userLink):
    if userStatus[idx] == 0:
        testX.append(data)
        testY.append(userGender[idx])
    else:
        trainX.append(data)
        trainY.append(userGender[idx])

clf = svm.SVC()
clf.fit(trainX, trainY)
predict = clf.predict(testX)
accuracy = list(predict == np.array(testY))
print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))

# Attack with Block modeling (BLOCK)
print("\n#### BLOCK ####")
linkCount = [[0, 0], [0, 0]]
for idx in range(len(userLink)):
    for idy in range(len(userLink)):
        if userLink[idx][idy] == 1:
            if userGender[idx] == 0 and userGender[idy] == 0:
                linkCount[0][0] += 1
            elif userGender[idx] == 0 and userGender[idy] == 1:
                linkCount[0][1] += 1
            elif userGender[idx] == 1 and userGender[idy] == 0:
                linkCount[1][0] += 1
            else:
                linkCount[1][1] += 1
# Generate lambda vector
linkCount = np.array(linkCount).astype(float)/np.sum(linkCount)
accuracy = []
for i in range(len(userLink)):
    userCount = [0, 0]
    # Do for only private user
    if userStatus[i] == 0:
        for j in range(len(userLink)):
            if userLink[i][j] == 1:
                if userGender[j] == 0:
                    userCount[0] += 1
                else:
                    userCount[1] += 1
        userCount = np.array(userCount).astype(float) / np.sum(userCount)
        # Find similarity with euclidean
        sim0 = np.linalg.norm(linkCount[0]-userCount)
        sim1 = np.linalg.norm(linkCount[1]-userCount)
        predictGender = 0 if sim0 > sim1 else 1
        accuracy.append(predictGender == userGender[i])
print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))

# Attack with Random guest (RAND)
print("\n#### RAND ####")
accuracy = []
for i in range(len(userLink)):
    # Do for only private user
    if userStatus[i] == 0:
        accuracy.append(np.random.randint(2) == userGender[i])
print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
