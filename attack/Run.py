import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
import sys
import glob
import os

sys.path.insert(0, '../sampling')
from extract_features import ExtractFeatures


f = open("./output/output_"+sys.argv[1]+"_"+sys.argv[2].replace("/", "_")+".csv","w")

for loop in range(int(sys.argv[3])):
    f.write("FILENAME, RAND, Pred, Act, RAND All, BASIC, Pred, Act, BASIC All, AGG, Pred, Act, AGG All, BLOCK, Pred, Act, BLOCK All, SVM, Pred, Act, SVM All, SVM Auc, NN, Pred, Act, NN All\n")

    for fileName in glob.glob("../sampling/output/"+sys.argv[2]+"/*"):
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


        # # Attack with Random guest (RAND)
        # print "RAND:  ",
        # accuracy = []
        # for i in range(0,len(userStatus)):
        #     if userStatus[i] == 0:
        #         predictGender = np.random.randint(numLabel)
        #         checkPredict = predictGender == np.argmax(userLabel[i])
        #         accuracy.append(checkPredict)
        #         if i == onlyTarget:
        #             print "{0:13}".format((checkPredict, predictGender, np.argmax(userLabel[i]))),
        #             f.write("," + str(checkPredict) + "," + str(predictGender) + "," + str(np.argmax(userLabel[i])))
        # print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        # f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        #
        #
        # # Attack without links (BASIC)
        # if loop == 0:
        #     print "BASIC: ",
        #     accuracy = []
        #     for target in range(0,len(userStatus)):
        #         if userStatus[target] == 0:
        #             countLabel = np.array([0 for i in range(numLabel)])
        #             countPublic = 0
        #             for idx, val in enumerate(userLabel):
        #                 if idx != target and userStatus[idx] != 0:
        #                     countLabel[np.argmax(val)] += 1
        #                     countPublic += 1
        #             checkPredict = np.argmax(countLabel) == np.argmax(userLabel[target])
        #             accuracy.append(checkPredict)
        #             if target == onlyTarget:
        #                 print "{0:13}".format((checkPredict, np.argmax(countLabel), np.argmax(userLabel[target]))),
        #                 f.write("," + str(checkPredict) + "," + str(np.argmax(countLabel)) + "," + str(np.argmax(userLabel[target])))
        #     print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        #     f.write("," + str(accuracy.count(True)/float(len(accuracy))))
        # else:
        #     f.write(",,,,")
        #
        #
        # # Attack with Friend-aggregate model (AGG)
        # if loop == 0:
        #     print "AGG:   ",
        #     accuracy = []
        #     for target in range(0,len(userStatus)):
        #         if userStatus[target] == 0:
        #             countLabel = np.array([0 for i in range(numLabel)])
        #             countPublic = 0
        #             for idx, val in enumerate(userLabel):
        #                 if idx != target and userStatus[idx] != 0 and userLink[idx,target] == 1:
        #                     countLabel[np.argmax(val)] += 1
        #                     countPublic += 1
        #             labelPredict = np.array([0 for i in range(numLabel)])
        #             labelPredict[np.argmax(countLabel)] = 1
        #             checkPredict = np.argmax(countLabel) == np.argmax(userLabel[target])
        #             accuracy.append(checkPredict)
        #             if target == onlyTarget:
        #                 print "{0:13}".format((checkPredict, np.argmax(countLabel), np.argmax(userLabel[target]))),
        #                 f.write("," + str(checkPredict) + "," + str(np.argmax(countLabel)) + "," + str(np.argmax(userLabel[target])))
        #     print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        #     f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        # else:
        #     f.write(",,,,")
        #
        #
        # # Attack with Block modeling (BLOCK)
        # if loop == 0:
        #     print "BLOCK: ",
        #     linkCount = [[0 for i in range(numLabel)] for i in range(numLabel)]
        #     for idx in range(len(userLink)):
        #         for idy in range(len(userLink)):
        #             if userLink[idx][idy] == 1:
        #                 linkCount[np.argmax(userLabel[idx])][np.argmax(userLabel[idy])] += 1
        #     linkCount = np.array(linkCount).astype(float)/np.sum(linkCount)
        #     accuracy = []
        #     for i in range(0,len(userStatus)):
        #         userCount = np.array([0 for k in range(numLabel)])
        #         if userStatus[i] == 0:
        #             for j in range(len(userLink)):
        #                 if userLink[i][j] == 1 and userStatus[j] == 1:
        #                     userCount[np.argmax(userLabel[j])] += 1
        #             if np.sum(userCount) != 0:
        #                 userCount = np.array(userCount).astype(float) / np.sum(userCount)
        #                 dis = [np.linalg.norm(linkCount[k]-userCount) for k in range(linkCount.shape[0])]
        #                 checkPredict = np.argmax(dis) == np.argmax(userLabel[i])
        #                 accuracy.append(checkPredict)
        #             if i == onlyTarget:
        #                 print "{0:13}".format((checkPredict, np.argmax(dis), np.argmax(userLabel[i]))),
        #                 f.write("," + str(checkPredict) + "," + str(np.argmax(dis)) + "," + str(np.argmax(userLabel[i])))
        #     print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        #     f.write("," + str(accuracy.count(True) / float(len(accuracy))))
        # else:
        #     f.write(",,,,")
        #
        # # Generate sampling data (under sampling)
        # trainX = []
        # trainY = []
        # testX = []
        # testY = []
        # for idx, data in enumerate(userLink):
        #     if userStatus[idx] == 0:
        #         testX.append(data)
        #         testY.append(np.argmax(userLabel[idx]))
        #         if idx == onlyTarget:
        #             newTarget = len(testX) - 1
        #     else:
        #         trainX.append(data)
        #         trainY.append(np.argmax(userLabel[idx]))
        # rus = RandomUnderSampler(return_indices=True)
        # xReSampled, yReSampled, idxReSampled = rus.fit_sample(trainX, trainY)
        #
        #
        # # Attack with SVM
        # print "SVM:   ",
        # clf = svm.SVC()
        # yScore = clf.fit(xReSampled, yReSampled).decision_function(testX)
        # predict = clf.predict(testX)
        # accuracy = list(predict == np.array(testY))
        # print "{0:13}".format((predict[newTarget] == testY[newTarget], predict[newTarget], testY[newTarget])),
        # f.write("," + str (predict[newTarget] == testY[newTarget]) + "," + str(predict[newTarget]) + "," + str(testY[newTarget]))
        # print " {0:13}".format((accuracy.count(True) / float(len(accuracy)))),
        # f.write("," + str (accuracy.count(True) / float(len(accuracy))))
        # print " {0}".format(roc_auc_score(testY, yScore))
        # f.write("," + str(roc_auc_score(testY, yScore)))
        #
        # # Attack with NN
        # print "NN:    ",
        # clf = MLPClassifier(activation='logistic',alpha=0.001, hidden_layer_sizes=(100,), max_iter=10000)
        # clf.fit(xReSampled, yReSampled)
        # predict = clf.predict(testX)
        # accuracy = list(predict == np.array(testY))
        # print "{0:13}".format((predict[newTarget] == testY[newTarget], predict[newTarget], testY[newTarget])),
        # f.write("," + str (predict[newTarget] == testY[newTarget]) + "," + str(predict[newTarget]) + "," + str(testY[newTarget]))
        # print " {0}".format((accuracy.count(True) / float(len(accuracy))))
        # f.write("," + str(accuracy.count(True) / float(len(accuracy))))

        # New line
        f.write("\n")
        print "\n"

    f.write(",\n")
    f.write(",\n")
f.close()
