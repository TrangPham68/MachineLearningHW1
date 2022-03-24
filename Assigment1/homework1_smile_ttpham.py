import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    """
    this takes in a vector of ground-truth labels and corresponding vector of guesses, and
    then computes the accuracy (PC)
    """
    return np.count_nonzero(np.equal(y, yhat.T), axis=1) / y.shape[0]

def measureAccuracyOfPredictors (predictors, X, y):
    """
    this takes in a set of predictors, a set of images
    to run it on, as well as the ground-truth labels of that set. For each image in the image set, it runs the
    ensemble to obtain a prediction. Then, it computes and returns the accuracy (PC) of the predictions
    w.r.t. the ground-truth labels.
    :param predictors:
    :param X:
    :param y:
    :return:
    """

    # pixel value predictors for each image
    pixelVal1 = X[:, predictors[:, :, 0], predictors[:, :, 1]]
    pixelVal2 = X[:, predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    #Using Mean Square error
    results = abs(pixelVal1 - pixelVal2)**2
    del pixelVal1
    del pixelVal2

    # calculate mean of all predictors
    avgResults = np.mean(results, axis=2)
    del results

    # convert from int to boolean (yes smiling no not smiling)
    results = np.greater(avgResults, np.full(avgResults.shape, .5))
    del avgResults

    return fPC(y, results)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    point = trainingFeature(trainingFaces, trainingLabels, testingFaces, testingLabels)
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for r1, c1, r2, c2 in point:
            # Show r1,c1
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        # Display the merged result
        plt.show()

def trainingFeature( trainingFaces, trainingLabels, testingFaces, testingLabels):
    # predictors, batch size, feature size
    bestPred = np.full((6, 4), None)
    #n = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    n = [2000]

    m = 6

    # find all permutations for 0-23
    idxs = np.arange(0, 24)
    allI = np.array(np.meshgrid(idxs, idxs, idxs, idxs)).T.reshape(-1, 4)

    for num in n:
        for i in range(m):
            allPred = np.full((len(allI), i + 1, 4), -1)

            # add the previous predictor to the list of new predictors
            allPred[:, :i, :] = bestPred[:i, :]

            # all possible new predictors
            allPred[:, i, :] = allI

            trainAcc = list()
            trainAcc.append(measureAccuracyOfPredictors(allPred, trainingFaces[:num], trainingLabels[:num]))

            # average our scores to get our final (training data) accuracy score
            #trainNum = np.array(trainAcc)
            trainNum = np.mean(trainAcc, axis=0)

            # locate our best predictor
            max = np.argmax(trainNum)

            # save our best predictor
            bestPred[i, :] = allPred[max, i, :]
            print(i)
        #acc = testAccuracy(np.array([allPred[max]]), trainingFaces, trainingLabels, testingFaces, testingLabels)
        #print(str(num) + " " + str(acc[0]) + " " + str(acc[1]))
        print(num)
        print(bestPred)
        print()
    return bestPred

def testAccuracy(bestPred, trainingFaces, trainingLabels, testingFaces, testingLabels):
    trainAcc = measureAccuracyOfPredictors(bestPred, trainingFaces, trainingLabels)
    testAcc = measureAccuracyOfPredictors(bestPred, testingFaces, testingLabels)
    return trainAcc, testAcc



def loadData (which):
    """Loading Data from test/train preset according to which type and reshape to 24x24"""
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)


