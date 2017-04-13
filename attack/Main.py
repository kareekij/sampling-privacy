import numpy as np
import csv
from sklearn import svm
import sys
import glob
from NnTf import NN
sys.path.insert(0, '../sampling')
from extract_features import ExtractFeatures

# # Connection between users
# userLink = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
#                      [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
#                      [0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
#                      [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
#                      [0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
#                      [1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
#                      [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
#                      [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
#                      [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
#                      [0, 0, 1, 0, 1, 0, 0, 1, 1, 0]])
#
# # Gender for each users
# userGender = []
# with open("testNetwork.txt") as csvFile:
#     readCSV = csv.reader(csvFile, delimiter="\t")
#     for row in readCSV:
#         userGender.append(int(row[3]))
#
# # Random 50% to be private
# a = [1] * 5
# b = [0] * 5
# userStatus = np.concatenate((a, b))
# np.random.shuffle(userStatus)

f = open("output.csv","w")
f.write("FILENAME, BASIC Target, BASIC, AGG Target, AGG, CC Target, CC, LINK Target, LINK, BLOCK Target, BLOCK, RAND Target, RAND, NN Target, NN\n")

for fileName in glob.glob("../sampling/output/*"):
    print "\n#################################"
    print "Processing: {0}".format(fileName)
    f.write(fileName)

    # Extract Feature
    fName = fileName
    e = ExtractFeatures(fName)
    nodeList = e.get_node_order()
    status = e.get_profile()

    # Label data that interest
    numLabel = 2
    gender = e.get_label('gender')

    # numLabel = 10
    # age = e.get_label('age')

    userLabel = []
    userStatus = []
    for node in nodeList:
        tmpLabel = [0 for i in range(numLabel)]
        tmpLabel[int(gender[node])] = 1
        # tmpLabel[int(age[node])%10] = 1
        userLabel.append(tmpLabel)
        userStatus.append(status[node])

    # All information that need to attack
    userLink = np.array(e.get_adjacency_matrix())
    userLabel = np.array(userLabel)
    userStatus = np.array(userStatus)

    # For only target ID
    onlyTarget = nodeList.index(e.get_target_node()[0])

    # Attack without links (BASIC)
    print("#### BASIC ####")
    accuracy = []
    for target in range(0,len(userStatus)):
        # Do for only private user
        if userStatus[target] == 0:
            countLabel = np.array([0 for i in range(numLabel)])
            countPublic = 0
            for idx, val in enumerate(userLabel):
                if idx != target and userStatus[idx] != 0:
                    countLabel[np.argmax(val)] += 1
                    countPublic += 1
            # Make prediction
            checkPredict = np.argmax(countLabel) == np.argmax(userLabel[target])
            accuracy.append(checkPredict)
            # Predict only the Target
            if target == onlyTarget:
                print checkPredict
                f.write("," + str(checkPredict))
    print("Accuracy: {0}".format(accuracy.count(True)/float(len(accuracy))))
    f.write("," + str(accuracy.count(True)/float(len(accuracy))))

    # Attack with Friend-aggregate model (AGG)
    print("\n#### AGG ####")
    accuracy = []
    userLabelPredict = []
    for target in range(0,len(userStatus)):
        # Do for only private user
        if userStatus[target] == 0:
            countLabel = np.array([0 for i in range(numLabel)])
            countPublic = 0
            for idx, val in enumerate(userLabel):
                # Consider only public user
                if idx != target and userStatus[idx] != 0 and userLink[idx,target] == 1:
                    countLabel[np.argmax(val)] += 1
                    countPublic += 1
            # Create new label
            labelPredict = np.array([0 for i in range(numLabel)])
            labelPredict[np.argmax(countLabel)] = 1
            userLabelPredict.append(labelPredict)
            # Make prediction
            checkPredict = np.argmax(countLabel) == np.argmax(userLabel[target])
            accuracy.append(checkPredict)
            if target == onlyTarget:
                print checkPredict
                f.write("," + str(checkPredict))
        else:
            userLabelPredict.append(userLabel[target])
    print("Accuracy: {0}".format(accuracy.count(True)/float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    # Attack with Collective classification model (CC)
    print("\n#### CC ####")
    accuracy = []
    # Train collective classifier
    for j in range(0, 1000):
        userLabelPredictNew = []
        for target in range(0,len(userStatus)):
            # Do for only private user
            if userStatus[i] == 0:
                countLabel = np.array([0 for i in range(numLabel)])
                countPublic = 0
                for idx, val in enumerate(userLabel):
                    # Consider both private and public
                    if idx != target and userLink[idx, target] == 1:
                        countLabel[np.argmax(val)] += 1
                        countPublic += 1
                # Create new label
                labelPredict = np.array([0 for i in range(numLabel)])
                labelPredict[np.argmax(countLabel)] = 1
                userLabelPredictNew.append(labelPredict)
            else:
                userLabelPredictNew.append(userLabel[target])
        userLabelPredict = userLabelPredictNew
    for i in range(0, len(userStatus)):
        if userStatus[i] == 0:
            # Make prediction
            checkPredict = np.argmax(userLabelPredict[i]) == np.argmax(userLabel[i])
            accuracy.append(checkPredict)
            if i == onlyTarget:
                print checkPredict
                f.write("," + str(checkPredict))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    #Attack with Flat-link model (LINK) with SVM
    print("\n#### LINK ####")
    trainX = []
    trainY = []
    testX = []
    testY = []
    for idx, data in enumerate(userLink):
        if userStatus[idx] == 0:
            testX.append(data)
            testY.append(np.argmax(userLabel[idx]))
            if idx == onlyTarget:
                newTarget = len(testX)
        else:
            trainX.append(data)
            trainY.append(np.argmax(userLabel[idx]))
    # SVM Model
    clf = svm.SVC()
    clf.fit(trainX, trainY)
    predict = clf.predict(testX)
    accuracy = list(predict == np.array(testY))
    print predict[newTarget] == testY[newTarget]
    f.write("," + str (predict[newTarget] == testY[newTarget]))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str (accuracy.count(True) / float(len(accuracy))))

    # Attack with Block modeling (BLOCK)
    print("\n#### BLOCK ####")
    linkCount = [[0 for i in range(numLabel)] for i in range(numLabel)]
    for idx in range(len(userLink)):
        for idy in range(len(userLink)):
            if userLink[idx][idy] == 1:
                linkCount[np.argmax(userLabel[idx])][np.argmax(userLabel[idy])] += 1
    # Generate lambda vector
    linkCount = np.array(linkCount).astype(float)/np.sum(linkCount)
    accuracy = []
    for i in range(0,len(userStatus)):
        userCount = np.array([0 for k in range(numLabel)])
        # Do for only private user
        if userStatus[i] == 0:
            for j in range(len(userLink)):
                if userLink[i][j] == 1 and userStatus[j] == 1:
                    userCount[np.argmax(userLabel[j])] += 1
            userCount = np.array(userCount).astype(float) / np.sum(userCount)
            # Find similarity with euclidean
            dis = [np.linalg.norm(linkCount[k]-userCount) for k in range(linkCount.shape[0])]
            # Make prediction
            checkPredict = np.argmax(dis) == np.argmax(userLabel[i])
            accuracy.append(checkPredict)
            if i == onlyTarget:
                print checkPredict
                f.write("," + str(checkPredict))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    # Attack with Random guest (RAND)
    print("\n#### RAND ####")
    accuracy = []
    for i in range(0,len(userStatus)):
        # Do for only private user
        if userStatus[i] == 0:
            predictGender = np.random.randint(numLabel)
            checkPredict = predictGender == np.argmax(userLabel[i])
            accuracy.append(checkPredict)
            if i == onlyTarget:
                print checkPredict
                f.write("," + str(checkPredict))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    # Attack with Link + Properties (NN)
    print("\n#### NN ####")
    trainX = []
    trainY = []
    testX = []
    testY = []
    for idx, data in enumerate(userLink):
        if userStatus[idx] == 0:
            testX.append(data)
            testY.append(userLabel[idx])
            if idx == onlyTarget:
                newTarget = len(testX)
        else:
            trainX.append(data)
            trainY.append(userLabel[idx])

    nn = NN(len(trainX[0]), len(trainY[0]))
    nn.fit(trainX, trainY, 1000)
    predict = nn.predict(testX)
    accuracy = list(predict == np.argmax(np.array(testY), 1))
    print predict[newTarget] == np.argmax(np.array(testY), 1)[newTarget]
    f.write("," + str(predict[newTarget] == np.argmax(np.array(testY), 1)[newTarget]))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))
    f.write("\n")
f.close()
