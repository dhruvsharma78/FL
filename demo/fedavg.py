from FL.optimizers import SGD
from FL.models import LogisticRegression
from sklearn.preprocessing import normalize
import numpy as np
import cvxpy as cvx
import legacy.classifier as classifier
import sys
import time
from mpi4py import MPI


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":


    # set up mpi stuff
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()


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
    num_coms = 10


    # split the data
    num_split = int(N//(p-1))

    # Solve using CVX to find optimal solution.
    w_star = cvx.Variable((dim, 1))
    loss = 1/N * cvx.sum(cvx.logistic(-cvx.multiply(y, X @ w_star))) + reg/2 * cvx.sum_squares(w_star)
    problem = cvx.Problem(cvx.Minimize(loss))
    problem.solve(verbose = False, abstol = 1e-15)
    opt = problem.value

    # server
    
    if my_rank == 0:
        
        y_start = time.process_time_ns()
        obj = np.zeros((exp, 1))
        MSE = np.zeros((exp, 1))
        w_aggregate = np.zeros_like(w_star.value)
        #w_aggregate = np.zeros((dim,1))
        
        for i in range(exp):
            # w_aggregate = np.zeros((dim,1))
            w_aggregate = np.zeros_like(w_star.value)
            for k in range(1, p):
                
                new_obj,new_w = comm.recv(source=k)
                #new_w = comm.recv(source=k)
                w_aggregate += new_w
                obj[i] += new_obj
            w_aggregate /= p-1
            for k in range(1, p):
                comm.send(w_aggregate, dest=k)
            MSE[i] = np.linalg.norm(w_aggregate- w_star.value, 2)
            obj[i] /= p-1


        print("Number of MPI nodes: {}".format(p - 1))
        print("Dataset size: {}".format((N, dim)))
        print("Time taken: {:.2f}s".format((time.process_time_ns() - y_start) /1000000000))
        print("Final result is " + str(obj[exp-1]))
        print("Final Weights:")
        print(w_aggregate)
        sys.stdout.flush()

    else:
    
        w_next = None
        X_split = X[(my_rank-1)*num_split:(my_rank)*num_split, :]
        y_split = y[(my_rank-1)*num_split:(my_rank)*num_split]
        for i in range(exp):
            optimizer = SGD(step_size=ss, reg=reg, epochs=ep, batch_size=batch, experiments=exp, decay=decay, w_next=w_next)
            model = LogisticRegression(optimizer)
            #obj,MSE,new_w = model.fit(X_split, y_split.reshape(num_split), w_star.value.reshape(dim), local=True, communications=num_coms, workers=5)
            obj,MSE,new_w = model.fit(X_split, y_split.reshape(num_split), w_star.value.reshape(dim), local=True, communications=num_coms, workers=5)
            avg_obj = np.average(obj)
            #comm.send((obj,new_w), dest=0)
            new_w_resized = np.zeros_like(w_star.value)
            
            for j in range(dim):
                new_w_resized[j] = new_w[j]
            
            comm.send((avg_obj, new_w_resized), dest=0)
            w_next = np.reshape(comm.recv(source=0), dim)
