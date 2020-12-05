import numpy as np

class Classifier():
    def __init__(self, w_star):
        """
        Member Variables
        - w (np.array) := Parameter of size (dim, 1).
        - w_star (np.array) := Actual parameters of size (dim, 1)
        """
        self.w = None
        self.w_star = w_star
    
    def obj(self, X, y, reg):
        """
        Compute the value of the loss function.
        Inputs:
        - X (np.array) := The data to calculate the loss on of
        size (N, dim).
        - y (np.array) := The corresponding labels of the data of size
        (N, 1).
        - reg (float) := The regularization parameter.
        Outputs: 
        - obj (float) := Value of the objective function.
        """
        pass
    
    def grad(self, X_batch, y_batch, reg):
        """
        Compute the gradient.
        Inputs:
        - X_batch (np.array) := The data to calculate the gradient on of
        size (N, dim).
        - y (np.array) := The corresponding labels of the data of size
        (N, 1).
        - reg (float) := The regularization parameter.
        Outputs:
        - grad (np.array) := Gradient with respect to self.w of size (dim, 1).
        """
        pass

    def SGD(self, X, y, step_size=0.1, reg=1e-4, epochs=100, batch_size=50, 
            experiments=1, w_next=None, decay=False):
        """
        Perform SGD.
        Input:
        - X (np.array) :=  The data of size (N x dim).
        - y (np.array) :=  The labels of the data of size (N x 1).
        - step_size (float) := Step size.
        - reg (float) := The regularization parameter.
        - epochs (int) := The number of passes through the data.
        - batch_size (int) := The amount of data to sample at each sub-iteration.
        - w_next (np.array) := If this is passed, we initialize w as w_next. It is of
        size (dim x 1).
        - decay (boolean) := False if no decay rate. True if use decaying rate.
        Output:
        - obj_SGD (np.array) := Array of objective values at each epoch. Has size (Epoch x 1)
        - obj_SGD_iters (np.array) := Array of objective values at each iteration. Has size (Epoch x 1)
        - MSE (np.array) := Array of the MSE at each iteration. (Iterations x 1)
        """
        N, dim = X.shape
        
        obj_SGD = np.zeros((epochs, 1))
        max_iters = int(N/batch_size)
        obj_SGD_iters = np.zeros((int(epochs*max_iters), 1))
        MSE = np.zeros((int(epochs*max_iters), 1))

        for k in range(experiments):
            self.w = 0.001 * np.random.randn(dim, 1)
            if w_next is not None:
                self.w = w_next
            for i in range(epochs):
                obj_SGD[i] += self.obj(X, y, reg).item()
                if i % 10 == 0:
                    print(f"Experiment: {k}/{experiments}, Epoch: {i}/{epochs}, Loss: {obj_SGD[i]/(k+1)}")
                for j in range(max_iters):
                    rand_idx = np.random.randint(0, N-1, batch_size)
                    X_batch = X[rand_idx, :]
                    y_batch = y[rand_idx]
                    obj_SGD_iters[i*max_iters + j] += self.obj(X, y, reg).item()
                    MSE[i*max_iters + j] += np.linalg.norm(self.w - self.w_star, 2)
                    if decay == False:
                        self.w = self.w - step_size * self.grad(X_batch, y_batch, reg)
                    else:
                        self.w = self.w - .1/(reg*(j+1000)) * self.grad(X_batch, y_batch, reg)
        obj_SGD /= experiments
        obj_SGD_iters /= experiments
        MSE /= experiments
        return obj_SGD, obj_SGD_iters, MSE

    def FedAVG(self, X, y, comm, step_size=0.1, reg=1e-4, epochs=100, batch_size=50, 
    experiments=25, w_next=None, decay=False, communications=100):
        """
        Perform FedAVG.
        Input:
        - X (np.array) := The data of size (N x dim).
        - y (np.array) :=  The labels of the data of size (N x 1).
        - comm (MPI.COMM_WORLD) := Variable derived from MPI4PY to handle parallelization. 
        - step_size (float) := Step size.
        - reg (float) := The regularization parameter.
        - epochs (int) := The number of passes through the data.
        - batch_size (int) := The amount of data to sample at each sub-iteration.
        - experiments (int) := Number of experiments to average over.
        - w_next (np.array) := If this is passed, we initialize w as w_next. It is of
        size (dim x 1).
        - decay (boolean) := False if no decay rate. True if use decaying rate.
        - communications (int) := Number of communication steps performed.
        Output:
        - obj_SGD (np.array) := Array of objective values at each communication. Has size (communications x 1)
        - MSE (np.array) := Array of the MSE at each communication. (communications x 1)
        """
        my_rank = comm.Get_rank()
        p = comm.Get_size()
        N, dim = X.shape
        num_split = int(N//(p-1))

        if my_rank == 0:
            # Initialize arrays to hold the results from each experiment.
            obj_SGD = np.zeros((communications, 1))
            MSE_SGD = np.zeros((communications, 1))

            for i in range(experiments):
                for j in range(communications):
                    # At every communication step receive all the parameters from the workers and average. 
                    w_aggregate_SGD = np.zeros_like(self.w_star) 
                    for k in range(1, p):
                        w_aggregate_SGD += comm.recv(source=k)
                    w_aggregate_SGD /= p-1
                    for k in range(1, p):
                        comm.send(w_aggregate_SGD, dest=k)
                    self.w = w_aggregate_SGD
                    obj_SGD[j] += self.obj(X, y, reg).item()
                    MSE_SGD[j] += np.linalg.norm(self.w - self.w_star, 2)
                    print(f"SGD, Experiment: {i} Communications: {j} Loss: {obj_SGD[j]/(i+1)}") 
            obj_SGD /= experiments
            MSE_SGD = 10 * np.log(MSE_SGD/experiments)
            return obj_SGD, MSE_SGD                 
        else:

            X_split = X[(my_rank-1)*num_split:(my_rank)*num_split, :]
            y_split = y[(my_rank-1)*num_split:(my_rank)*num_split]
            for i in range(experiments):
                if w_next == None:
                    w_global_SGD = 0.001 * np.random.randn(dim, 1)
                else:
                    w_global_SGD = w_next
                for j in range(communications):
                    self.SGD(X_split, y_split, step_size=step_size, reg=reg, epochs=epochs, batch_size=batch_size, w_next=w_global_SGD, decay=decay)
                    comm.send(self.w, dest=0)
                    w_global_SGD = comm.recv(source=0)
            pass

class LogReg(Classifier):
    def __init__(self, w_star):
        super(LogReg, self).__init__(w_star)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def obj(self, X, y, reg):
        N, _ = X.shape
        return 1/N * np.sum(np.log(1 + np.exp(-y * X @ self.w))) + 1/2 * reg * self.w.T @ self.w 
    
    def grad(self, X_batch, y_batch, reg):
        N_batch, _ = X_batch.shape
        return 1/N_batch * X_batch.T @ (y_batch * (self.sigmoid(y_batch * X_batch @ self.w) - 1)) + reg * self.w 









    

        
        
