# Federated Learning 

# Setup Windows (Only works with Windows right now. I'll update environment later.): 

1) Install Microsoft MPI v10.0 from https://www.microsoft.com/en-us/download/details.aspx?id=57467&WT.mc_id=rss_alldownloads_devresources. Make sure to add C:\Program Files\Microsoft MPI\Bin\ to your "Environment Variables".

2) This project uses Anaconda environments to guarantee package management. There is a .yml file provided. Open the command prompt. Please build a conda environment from this file using the command "conda env create -n FLRR -f windows_environment.yml". Activate the environment using "conda activate FLRR". Verify that the new environment is working using conda env list.

# Usage:

1) With the conda environment activated use the command "mpiexec -n 5 python fedavg_demo.py"

2) 5 can be replaced with any number of processes. Performance will be based on the number of cores of the computer. At the end of the simulation, a picture of the MSE performance will be generated. Note that process "0" or as in the code "my_rank = 0" denotes the server. Hence, when we use 5 processes we will have 1 server and 4 workers.

3) If one wants to run a different simulation, copy the fedavg_demo.py file.

4) Open the .py file and see the different parameters that can be modified such as the number of experiments, dimension of a data point, etc.

5) Currently usage is limited to l2-regularized logisitic regression. We can extend the functionality to different types of problem by creating a child-class from the base class Classifier and providing an objective/gradient function. See the details in the classifier.py file.

