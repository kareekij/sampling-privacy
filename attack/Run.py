import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sys
import glob
import os

sys.path.insert(0, '../sampling')
from extract_features import ExtractFeatures


f = open("./" + sys.argv[5] +"/output_"+sys.argv[1]+"_"+sys.argv[2].replace("/", "_")+".csv","w")

for loop in range(int(sys.argv[4])):
    # f.write("FILENAME, RAND, RAND.Pred, RAND.Act, RAND.All, BASIC, BASIC.Pred, BASIC.Act, BASIC.All, AGG, AGG.Pred, AGG.Act, AGG.All, BLOCK, BLOCK.Pred, BLOCK.Act, BLOCK.All, SVMR, SVMR.Pred, SVMR.Act, SVMR.All, NN, NN.Pred, NN.Act, NN.All,
    # NB, NB.Pred, NB.Act, NB.All, DT, DT.Pred, DT.Act, DT.All, RDF, RDF.Pred, RDF.Act, RDF.All, ABD, ABD.Pred, ABD.Act, ABD.All, SVML, SVML.Pred, SVML.Act, SVML.All,
    # SVMP, SVMP.Pred, SVMP.Act, SVMP.All, SVMS, SVMS.Pred, SVMS.Act, SVMS.All, Folder, Category, Trial\n")

    for fileName in glob.glob("../sampling/" + sys.argv[5] +"/"+sys.argv[2]+"/*"):
        print "Processing:      {0}".format(os.path.basename(fileName))
        f.write(os.path.basename(fileName))

        # Read data from pickle file and put into format
        fName = fileName
        e = ExtractFeatures(fName)
        nodeList = e.get_node_order()
        status = e.get_profile()
        degree = e.get_degree()
        coefficient = e.get_clustering_coefficient()
        userLabel = []
        userStatus = []
        userDegree = []
        userCoeff = []
        numLabel = 2
        gender = e.get_label('gender')
        for node in nodeList:
            tmpLabel = [0 for i in range(numLabel)]
            tmpLabel[int(gender[node])] = 1
            userLabel.append(tmpLabel)
            userStatus.append(status[node])
            userDegree.append(degree[node])
            userCoeff.append(coefficient[node])
        userLink = np.array(e.get_adjacency_matrix()).astype(np.float32)
        userLabel = np.array(userLabel).astype(np.float32)
        userStatus = np.array(userStatus).astype(np.float32)
        onlyTarget = nodeList.index(e.get_target_node()[0])
        print "Distribution:    {0}".format(np.sum(userLabel, axis=0))
        print "TargetID, TargetIndex, Label: {0}".format((e.get_target_node()[0], onlyTarget, np.argmax(userLabel[onlyTarget])))


        # Attack with Random guest (RAND)
        print "RAND:  ",
        accuracy = []
        for i in range(0,len(userStatus)):
            if userStatus[i] == 0:
                predictGender = np.random.randint(numLabel)
                checkPredict = predictGender == np.argmax(userLabel[i])
                accuracy.append(checkPredict)
                if i == onlyTarget:
                    print "{0:13}".format((int(checkPredict), predictGender, np.argmax(userLabel[i]))),
                    f.write("," + str(int(checkPredict)) + "," + str(predictGender) + "," + str(np.argmax(userLabel[i])))
        print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))


        # Attack without links (BASIC)
        if loop == 0:
            print "BASIC: ",
            accuracy = []
            publicProfile = np.where(userStatus == 1)
            countLabel = np.sum(userLabel[publicProfile],axis=0)
            for target in range(0,len(userStatus)):
                if userStatus[target] == 0:
                    checkPredict = np.argmax(countLabel) == np.argmax(userLabel[target])
                    accuracy.append(checkPredict)
                    if target == onlyTarget:
                        print "{0:13}".format((int(checkPredict), np.argmax(countLabel), np.argmax(userLabel[target]))),
                        f.write("," + str(int(checkPredict)) + "," + str(np.argmax(countLabel)) + "," + str(np.argmax(userLabel[target])))
            print " {0}".format((accuracy.count(True) / float(len(accuracy))))
            f.write("," + str(accuracy.count(True)/float(len(accuracy))))
        else:
            f.write(",,,,")


        # Attack with Friend-aggregate model (AGG)
        if loop == 0:
            print "AGG:   ",
            accuracy = []
            publicProfile = np.where(userStatus == 1)
            for target in range(0,len(userStatus)):
                if userStatus[target] == 0:
                    countLabel = np.sum(userLabel[np.intersect1d(np.where(userLink[target] == 1), publicProfile)],axis=0)
                    checkPredict = np.argmax(countLabel) == np.argmax(userLabel[target])
                    accuracy.append(checkPredict)
                    if target == onlyTarget:
                        print "{0:13}".format((int(checkPredict), np.argmax(countLabel), np.argmax(userLabel[target]))),
                        f.write("," + str(int(checkPredict)) + "," + str(np.argmax(countLabel)) + "," + str(np.argmax(userLabel[target])))
            print " {0}".format((accuracy.count(True) / float(len(accuracy))))
            f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        else:
            f.write(",,,,")


        # Attack with Block modeling (BLOCK)
        if loop == 0:
            print "BLOCK: ",
            linkCount = [[0 for i in range(numLabel)] for i in range(numLabel)]
            userLinkIndex = np.where(userLink == 1)
            for i in range(len(userLinkIndex[0])):
                source = userLinkIndex[0][i]
                des = userLinkIndex[1][i]
                if userStatus[source] == 1 and userStatus[des] == 1:
                    linkCount[np.argmax(userLabel[source])][np.argmax(userLabel[des])] += 1
            linkCount = np.array(linkCount).astype(float)/np.sum(linkCount)
            accuracy = []
            publicProfile = np.where(userStatus == 1)
            for i in range(0,len(userStatus)):
                if userStatus[i] == 0:
                    userCount = np.sum(userLabel[np.intersect1d(np.where(userLink[i] == 1), publicProfile)],axis=0)
                    if np.sum(userCount) != 0:
                        userCount = np.array(userCount).astype(float) / np.sum(userCount)
                        dis = [np.linalg.norm(linkCount[k]-userCount) for k in range(linkCount.shape[0])]
                        checkPredict = np.argmax(dis) == np.argmax(userLabel[i])
                        accuracy.append(checkPredict)
                    if i == onlyTarget:
                        print "{0:13}".format((int(checkPredict), np.argmax(dis), np.argmax(userLabel[i]))),
                        f.write("," + str(int(checkPredict)) + "," + str(np.argmax(dis)) + "," + str(np.argmax(userLabel[i])))
            print " {0}".format((accuracy.count(True) / float(len(accuracy))))
            f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        else:
            f.write(",,,,")

        # Generate sampling data (under sampling)
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
        xReSampled, yReSampled, idxReSampled = rus.fit_sample(trainX, trainY)


        # Attack with SVM RBF
        print "SVMR:  ",
        clf = svm.SVC(kernel='rbf')
        yScore = clf.fit(xReSampled, yReSampled).decision_function(testX)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(testY[newTarget]))
        print " {0:13}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str (accuracy.count(True) / float(len(accuracy))))
        # print " {0}".format(roc_auc_score(testY, yScore))
        # f.write("," + str(roc_auc_score(testY, yScore)))


        # Attack with NN
        print "NN:    ",
        clf = MLPClassifier(activation='logistic',alpha=0.001, hidden_layer_sizes=(100,), max_iter=10000)
        clf.fit(xReSampled, yReSampled)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(testY[newTarget]))
        print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))


        # Attack with Naive Bay
        print "NB:    ",
        clf = GaussianNB()
        clf.fit(xReSampled, yReSampled)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))


        # Attack with Decision Tree
        print "DT:    ",
        clf = tree.DecisionTreeClassifier()
        clf.fit(xReSampled, yReSampled)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))


        # Attack with Random Forest
        print "RDF:   ",
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(xReSampled, yReSampled)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))


        # Attack with AdaBoost
        print "ADB:   ",
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(xReSampled, yReSampled)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))


        # Attack with SVM Linear
        print "SVML:  ",
        clf = svm.SVC(kernel='linear')
        yScore = clf.fit(xReSampled, yReSampled).decision_function(testX)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0:13}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        # print " {0}".format(roc_auc_score(testY, yScore))
        # f.write("," + str(roc_auc_score(testY, yScore)))


        # Attack with SVM Poly
        print "SVMP:  ",
        clf = svm.SVC(kernel='poly')
        yScore = clf.fit(xReSampled, yReSampled).decision_function(testX)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0:13}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        # print " {0}".format(roc_auc_score(testY, yScore))
        # f.write("," + str(roc_auc_score(testY, yScore)))


        # Attack with SVM Sigmoid
        print "SVMS:  ",
        clf = svm.SVC(kernel='sigmoid')
        yScore = clf.fit(xReSampled, yReSampled).decision_function(testX)
        predict = clf.predict(testX)
        accuracy = list(predict == np.array(testY))
        print "{0:13}".format((int(predict[newTarget] == testY[newTarget]), predict[newTarget], testY[newTarget])),
        f.write("," + str(int(predict[newTarget] == testY[newTarget])) + "," + str(predict[newTarget]) + "," + str(
            testY[newTarget]))
        print " {0:13}".format((accuracy.count(True) / float(len(accuracy))))
        f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        # print " {0}".format(roc_auc_score(testY, yScore))
        # f.write("," + str(roc_auc_score(testY, yScore)))


        # New line
        f.write("," + sys.argv[6] + "," + sys.argv[3] + "," + str(loop))
        f.write("\n")
        print "\n"

    # f.write(",\n")
    # f.write(",\n")
f.close()
