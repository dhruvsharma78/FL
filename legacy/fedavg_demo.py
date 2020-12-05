from mpi4py import MPI
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cvx
import classifier 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    """
    Global Parameters:
        N (int):= Number of data points.
        dim (int) := Dimension of a data point.
        reg (float) := regularization parameter
        w (np.array) := Parameters of size (dim, 1).
        X (np.array) := Data of size (N, dim).
        y (np.array) := Labels of data of size (dim, 1)
        ss (float) := Step size.
        ep (int) := Number of epochs.
        batch (int) := Batch size.
        exp (int) := Number of experiments.
        decay (boolean) := Whether to use decay or not.
        num_comms (int) := Communications is how many times we send our information to the server 
        for syncing.
    """
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()

    # Assume that the number of the data splits evenly among each workers.

    N = 8000
    dim = 10
    reg = 1e-4
    np.random.seed(0)
    w = np.random.multivariate_normal([0.0]*dim, np.eye(dim), 1).T
    X = np.random.multivariate_normal([0.0]*dim, np.eye(dim), size = N)
    X = normalize(X, axis = 1, norm = 'l2')
    y = 2 * (np.random.uniform(size = (N, 1)) < sigmoid(X @ w)) - 1

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
    if my_rank == 0:
        print(f'Optimal objective function value is: {opt}')

    solver = classifier.LogReg(w_star.value)

    if my_rank == 0:
        obj_SGD, MSE_SGD = solver.FedAVG(X, y, comm, step_size=ss, reg=reg, epochs=ep, batch_size=batch, experiments=exp, decay=decay, communications=num_coms)
        iters = np.arange(len(MSE_SGD))
        MSE_fig, ax = plt.subplots()
        ax.plot(iters, MSE_SGD, label = 'FedAVG')
        ax.set_title('MSE of FedAVG')
        ax.legend(loc = 'upper right')
        ax.set_ylabel('MSE (dB)')
        ax.set_xlabel('Communications')
        plt.savefig("Figures/MSE_FedAVG_constant_SS_fig")
        plt.show()
    else:
        solver.FedAVG(X, y, comm, step_size=ss, reg=reg, epochs=ep, batch_size=batch, experiments=exp, decay=decay, communications=num_coms)
