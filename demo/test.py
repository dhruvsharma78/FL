from FL.optimizers import SGD
from FL.models import LogisticRegression
from FL.util import math as flm
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cvx
import legacy.classifier as classifier
import sys
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

    N = 8000
    dim = 10
    reg = 1e-4
    np.random.seed(0)
    w = np.random.multivariate_normal([0.0]*dim, np.eye(dim), 1).T
    X = np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N)
    X = normalize(X, axis = 1, norm = 'l2')
    y = 2.0 * (np.random.uniform(size = (N, 1)) < sigmoid(X @ w)) - 1

    ss = .1
    ep = 5
    batch = 50
    exp = 25
    decay = False
    num_coms = 100

    # Solve using CVX to find optimal solution.
    w_star = cvx.Variable((dim, 1))
    loss = 1/N * cvx.sum(cvx.logistic(-cvx.multiply(y, X @ w_star))) + reg/2 * cvx.sum_squares(w_star)
    problem = cvx.Problem(cvx.Minimize(loss))
    problem.solve(verbose = False, abstol = 1e-15)
    opt = problem.value

    if sys.argv[1] == "p":
        solver = classifier.LogReg(w_star.value)
        obj_SGD, _, MSE_SGD = solver.SGD(X, y, step_size=ss, reg=reg, epochs=ep, batch_size=batch, experiments=exp, decay=decay)
        # obj_SGD, MSE_SGD = solver.FedAVG(X, y, comm, step_size=ss, reg=reg, epochs=ep, batch_size=batch, experiments=exp, decay=decay, communications=num_coms)
        print(solver.w)
        iters = np.arange(len(MSE_SGD))
        MSE_fig, ax = plt.subplots()
        ax.plot(iters, MSE_SGD, label = 'FedAVG')
        ax.set_title('MSE of FedAVG')
        ax.legend(loc = 'upper right')
        ax.set_ylabel('MSE (dB)')
        ax.set_xlabel('Communications')
        plt.show()
    elif sys.argv[1] == "c":
        optimizer = SGD(step_size=ss, reg=reg, epochs=ep, batch_size=batch, experiments=exp, decay=decay)
        model = LogisticRegression(optimizer)
        y_start = time.process_time_ns()
        # _,_,MSE,x = model.fit(X, y.reshape(N), w_star.value.reshape(dim))
        _,MSE,x = model.fit(X, y.reshape(N), w_star.value.reshape(dim), local=True, communications=num_coms, workers=5)
        print("Time taken: {:.2f}s".format((time.process_time_ns() - y_start) /1000000000))
        print(x.reshape((dim, 1)))
        iters = np.arange(len(MSE))
        MSE_fig, ax = plt.subplots()
        ax.plot(iters, MSE, label = 'FedAVG')
        ax.set_title('MSE of FedAVG')
        ax.legend(loc = 'upper right')
        ax.set_ylabel('MSE (dB)')
        ax.set_xlabel('Communications')
        plt.show()
    else:
        print("Usage: python test.py [pc]\nCompare performance of C version with pure python version")
