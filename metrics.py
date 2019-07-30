import numpy as np
from sklearn.metrics import f1_score


def accuracy(targets, predicted):
    accuracies = []
    for target, prediction in zip(targets, predicted):
        prediction = np.array(prediction)
        target = np.array(target)
        accuracies.append(np.sum(prediction == target) / float(target.size))
    return np.mean(accuracies)

def f1(targets, predicted, threshold=0.5):
    f1s = []
    for target, prediction in zip(targets, predicted):
        target = np.array(target)
        prediction = np.array(prediction)
        prediction = (prediction >= threshold).astype(np.int32)
        score = f1_score(target.flatten(), prediction.flatten())
        f1s.append(score)
    return np.mean(f1s)
