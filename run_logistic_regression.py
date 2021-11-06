from q3.check_grad import check_grad
from q3.utils import *
from q3.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # ADDED:                                                            #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.01,  # small dataset: 0.5
        "weight_regularization": 0.,
        "num_iterations": 770  # small dataset: 900
    }
    weights = [[0] for _ in range(M + 1)]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # ADDED:                                                            #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    lst_ce_train, lst_ce_val = np.array([]), np.array([])
    for t in range(hyperparameters["num_iterations"]):
        # compute cross entropy loss, derivative of the loss function w.r.t. weights, prediction labels
        f_train, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        f_val, _df, _y = logistic(weights, valid_inputs, valid_targets, hyperparameters)

        # update rule
        weights -= hyperparameters["learning_rate"] * df

        # store the average cross entropy loss for each iteration
        lst_ce_train = np.append(lst_ce_train, f_train)
        lst_ce_val = np.append(lst_ce_val, f_val)
    classifier_output_train = logistic_predict(weights, train_inputs)
    classifier_output_val = logistic_predict(weights, valid_inputs)
    classifier_output_test = logistic_predict(weights, test_inputs)
    ce_train, cr_train = evaluate(train_targets, classifier_output_train)
    ce_val, cr_val = evaluate(valid_targets, classifier_output_val)
    ce_test, cr_test = evaluate(test_targets, classifier_output_test)
    print(f"Averaged Cross Entropy of the Training Set: {round(ce_train, 3)}")
    print(f"Averaged Cross Entropy of the Validation Set: {round(ce_val, 3)}")
    print(f"Averaged Cross Entropy of the Testing Set: {round(ce_test, 3)}")
    print(f"Classification Rate of the Training Set: {round(cr_train, 3)}")
    print(f"Classification Rate of the Validation Set: {round(cr_val, 3)}")
    print(f"Classification Rate of the Testing Set: {round(cr_test, 3)}")
    print(f"Classification Error of the Training Set: {round(1.0 - cr_train, 3)}")
    print(f"Classification Error of the Validation Set: {round(1.0 - cr_val, 3)}")
    print(f"Classification Error of the Testing Set: {round(1.0 - cr_test, 3)}")

    # part (c): plot cross entropy for mnist_train and mnist_train_small
    fig, axes = plt.subplots()
    axes.plot(np.arange(0, hyperparameters["num_iterations"]), lst_ce_train, 'r', label="training")
    axes.plot(np.arange(0, hyperparameters["num_iterations"]), lst_ce_val, 'b', label="validation")
    axes.set(title="mnist_train_small: Average Cross Entropy w.r.t. # iterations")
    axes.set_xlabel("number of iterations")
    axes.set_ylabel("average cross entropy loss")
    plt.legend()
    # fig.savefig("Q3.2(c)_mnist_train_small.png")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
