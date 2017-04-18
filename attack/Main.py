import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
import sys
import glob

sys.path.insert(0, '../sampling')
from extract_features import ExtractFeatures

f = open("output_"+sys.argv[1]+"_"+sys.argv[2]+".csv","w")
f.write("FILENAME, BASIC Target, BASIC, AGG Target, AGG, CC Target, CC, LINK Target, LINK, BLOCK Target, BLOCK, RAND Target, RAND, NN Target, NN\n")

for fileName in glob.glob("../sampling/output/"+sys.argv[2]+"/*"):
    print "\n#################################"
    print "Processing: {0}".format(fileName)
    f.write(fileName)

    # Extract Feature
    fName = fileName
    e = ExtractFeatures(fName)
    nodeList = e.get_node_order()
    status = e.get_profile()
    degree = e.get_degree()
    coefficient = e.get_clustering_coefficient()

    # Label data that interest and read Data
    userLabel = []
    userStatus = []
    userDegree = []
    userCoeff = []
    if sys.argv[1] == "Gender":
        numLabel = 2
        gender = e.get_label('gender')
        for node in nodeList:
            tmpLabel = [0 for i in range(numLabel)]
            tmpLabel[int(gender[node])] = 1
            userLabel.append(tmpLabel)
            userStatus.append(status[node])
            userDegree.append(degree[node])
            userCoeff.append(coefficient[node])
    else:
        numLabel = 10
        age = e.get_label('age')
        for node in nodeList:
            tmpLabel = [0 for i in range(numLabel)]
            tmp = int(age[node])/
            classLabel = tmp if tmp < numLabel else numLabel-1
            tmpLabel[classLabel] = 1
            userLabel.append(tmpLabel)
            userStatus.append(status[node])
            userDegree.append(degree[node])
            userCoeff.append(coefficient[node])

    # All information that need to attack
    userLink = np.array(e.get_adjacency_matrix()).astype(np.float32)
    userLabel = np.array(userLabel).astype(np.float32)
    userStatus = np.array(userStatus).astype(np.float32)

    # For only target ID
    if len(e.get_target_node()) != 0:
        onlyTarget = nodeList.index(e.get_target_node()[0])
    else:
        onlyTarget = 0

    print np.sum(userLabel, axis=0)
    print (onlyTarget, np.argmax(userLabel[onlyTarget]))

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
                print (checkPredict, np.argmax(countLabel))
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
                print (checkPredict, np.argmax(countLabel))
                f.write("," + str(checkPredict))
        else:
            userLabelPredict.append(userLabel[target])
    print("Accuracy: {0}".format(accuracy.count(True)/float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    # # Attack with Collective classification model (CC)
    # print("\n#### CC ####")
    # accuracy = []
    # # Train collective classifier
    # for j in range(0, 1000):
    #     userLabelPredictNew = []
    #     for target in range(0,len(userStatus)):
    #         # Do for only private user
    #         if userStatus[target] == 0:
    #             countLabel = np.array([0 for i in range(numLabel)])
    #             countPublic = 0
    #             for idx, val in enumerate(userLabelPredict):
    #                 # Consider both private and public
    #                 if idx != target and userLink[idx, target] == 1:
    #                     countLabel[np.argmax(val)] += 1
    #                     countPublic += 1
    #             # Create new label
    #             labelPredict = np.array([0 for i in range(numLabel)])
    #             labelPredict[np.argmax(countLabel)] = 1
    #             userLabelPredictNew.append(labelPredict)
    #         else:
    #             userLabelPredictNew.append(userLabel[target])
    #     userLabelPredict = userLabelPredictNew
    # for i in range(0, len(userStatus)):
    #     if userStatus[i] == 0:
    #         # Make prediction
    #         checkPredict = np.argmax(userLabelPredict[i]) == np.argmax(userLabel[i])
    #         accuracy.append(checkPredict)
    #         if i == onlyTarget:
    #             print (checkPredict, np.argmax(userLabelPredict[i]))
    #             f.write("," + str(checkPredict))
    # print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    # f.write("," + str(accuracy.count(True) / float(len(accuracy))))

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
                newTarget = len(testX)-1
        else:
            trainX.append(data)
            trainY.append(np.argmax(userLabel[idx]))

    rus = RandomUnderSampler(return_indices=True)
    X_resampled, y_resampled, idx_resampled = rus.fit_sample(trainX, trainY)
    print [np.count_nonzero(y_resampled == label) for label in range(numLabel)]

    # SVM Model
    clf = svm.SVC()
    # clf.fit(trainX, trainY)
    clf.fit(X_resampled, y_resampled)
    predict1 = clf.predict(X_resampled)
    accuracy1 = list(predict1 == np.array(y_resampled))
    print("Training Accuracy: {0}".format(accuracy1.count(True) / float(len(accuracy1))))

    predict = clf.predict(testX)
    accuracy = list(predict == np.array(testY))
    print (predict[newTarget] == testY[newTarget], predict[newTarget])
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
            if np.sum(userCount) != 0:
                userCount = np.array(userCount).astype(float) / np.sum(userCount)
                # Find similarity with euclidean
                dis = [np.linalg.norm(linkCount[k]-userCount) for k in range(linkCount.shape[0])]
                # Make prediction
                checkPredict = np.argmax(dis) == np.argmax(userLabel[i])
                accuracy.append(checkPredict)
            if i == onlyTarget:
                print (checkPredict, np.argmax(dis))
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
                print (checkPredict, predictGender)
                f.write("," + str(checkPredict))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    # # Attack with Link + Properties (NN)
    # # userLink = np.insert(userLink, userLink.shape[1], userDegree, axis=1)
    # # userLink = np.insert(userLink, userLink.shape[1], userCoeff, axis=1)
    # print("\n#### NN ####")
    # trainX = []
    # trainY = []
    # testX = []
    # testY = []
    # for idx, data in enumerate(userLink):
    #     if userStatus[idx] == 0:
    #         testX.append(data)
    #         testY.append(userLabel[idx])
    #         if idx == onlyTarget:
    #             newTarget = len(testX)-1
    #     else:
    #         trainX.append(data)
    #         trainY.append(userLabel[idx])
    #
    # # testX = np.array(testX)
    # # trainX = np.array(trainX)
    # # mean = trainX.mean(axis=0)
    # # std = trainX.std(axis=0)
    # # trainX[:, trainX.shape[1]-2:trainX.shape[1]] = (trainX[:, trainX.shape[1]-2:trainX.shape[1]]-mean[trainX.shape[1]-2:trainX.shape[1]])/std[trainX.shape[1]-2:trainX.shape[1]]
    # # testX[:, testX.shape[1] - 2:testX.shape[1]] = (testX[:, testX.shape[1] - 2:testX.shape[1]]-mean[testX.shape[1] - 2:testX.shape[1]])/std[testX.shape[1] - 2:testX.shape[1]]
    # # trainX = (trainX-mean)/std
    # # testX = (testX-mean)/std
    #
    # print (len(userLink), len(trainX), len(testX), len(userLink) - len(trainX) - len(testX))
    # nn = NN(len(trainX[0]), len(trainY[0]))
    # nn.fit(trainX, trainY, 5000)
    # predict = nn.predict(testX)
    # accuracy = list(predict == np.argmax(np.array(testY), 1))
    # print (predict[newTarget] == np.argmax(np.array(testY), 1)[newTarget], predict[newTarget])
    # f.write("," + str(predict[newTarget] == np.argmax(np.array(testY), 1)[newTarget]))
    # print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    # f.write("," + str(accuracy.count(True) / float(len(accuracy))))
    # f.write("\n")

    # # Attack with Link + Properties
    # print("\n#### LINK+PROP ####")
    # trainX = []
    # trainY = []
    # testX = []
    # testY = []
    # for idx, data in enumerate(userLink):
    #     if userStatus[idx] == 0:
    #         testX.append(data)
    #         testY.append(np.argmax(userLabel[idx]))
    #         if idx == onlyTarget:
    #             newTarget = len(testX)
    #     else:
    #         if userDegree[idx] > 4: #and userCoeff[idx] > 0:
    #             trainX.append(data)
    #             trainY.append(np.argmax(userLabel[idx]))
    #
    # # SVM Model
    # print (len(userLink), len(trainX), len(testX), len(userLink)-len(trainX)-len(testX))
    # clf = svm.SVC()
    # clf.fit(trainX, trainY)
    # predict = clf.predict(testX)
    # accuracy = list(predict == np.array(testY))
    # print predict[newTarget] == testY[newTarget]
    # f.write("," + str(predict[newTarget] == testY[newTarget]))
    # print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    # f.write("," + str(accuracy.count(True) / float(len(accuracy))))

    # Attack with Flat-link model (LINK) with SVM
    print("\n#### NN ####")
    trainX = []
    trainY = []
    testX = []
    testY = []
    for idx, data in enumerate(userLink):
        if userStatus[idx] == 0:
            testX.append(data)
            testY.append(np.argmax(userLabel[idx]))
            if idx == onlyTarget:
                newTarget = len(testX) - 1
        else:
            trainX.append(data)
            trainY.append(np.argmax(userLabel[idx]))

    rus = RandomUnderSampler(return_indices=True)
    X_resampled, y_resampled, idx_resampled = rus.fit_sample(trainX, trainY)
    print [np.count_nonzero(y_resampled == label) for label in range(numLabel)]

    # NN Model
    hiddenNeuron = (len(trainX[0])+max(trainY))/2
    clf1 = MLPClassifier(activation='logistic',alpha=0.001, hidden_layer_sizes=(100,), max_iter=10000)
    # clf1.fit(trainX, trainY)
    clf1.fit(X_resampled, y_resampled)
    predict1 = clf1.predict(X_resampled)
    accuracy1 = list(predict1 == np.array(y_resampled))
    print("Training Accuracy: {0}".format(accuracy1.count(True) / float(len(accuracy1))))

    predict = clf1.predict(testX)
    accuracy = list(predict == np.array(testY))
    print (predict[newTarget] == testY[newTarget], predict[newTarget])
    f.write("," + str(predict[newTarget] == testY[newTarget]))
    print("Accuracy: {0}".format(accuracy.count(True) / float(len(accuracy))))
    f.write("," + str(accuracy.count(True) / float(len(accuracy))))
    f.write("\n")
f.close()
