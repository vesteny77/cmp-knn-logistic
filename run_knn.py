from q3.l2_distance import l2_distance
from q3.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # ADDED:
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    lst_k = [1, 3, 5, 7, 9]
    predicted_labels_for_each_k = []  # 5 inner lists where each list is a prediction list for k = i
    for k in lst_k:
        predicted_labels_for_each_k.append(knn(k, train_inputs, train_targets, valid_inputs))
    classification_rates = [(np.sum(valid_targets == predicted_labels) / len(valid_targets))
                            for predicted_labels in predicted_labels_for_each_k]
    fig, axes = plt.subplots()
    axes.plot(lst_k, classification_rates, "b")
    axes.set_xlabel("k")
    axes.set_ylabel("Classification Rate")
    axes.set_title("Classification rates with respect k")
    plt.show()
    # fig.savefig("q3.1(a).png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
