# Federated Learning

# Usage

```
from FL.models import LogisticRegression
from FL.optimizers import SGD
...
optimizer = SGD(step_size=0,1,
                reg=0.001,
                epochs=100,
                batch_size=50,
                experiments=25,
                decay=True)
model = LogisticRegression(optimizer)

# Regular SGD
# X = array of dimensions N x dim
# y, w_star = vectors of dimension dim
obj, obj_iteration, MSE, x = model.fit(X, y, w_star, parallel=False)

# Local SGD
# Workers = number of concurrent threads
obj, MSE, x = model.fit(X, y, w_star, local=True, communications=100, workers=5)

```

# Installation

1. Install required packages listed in requirements.txt
2. Install OpenMP and Intel MKL
3. In setup.py change the global variables `extra_compile_args`, `extra_link_args`, `lib_paths` and `libs` to match your installation of MKL and OpenMP along with the compiler you wish to use. Use the [Intel MKL Link Line Advisor](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html) to find your link and compile arguments. The default setup works with `gcc` and uses the TBB threaded version of Intel MKL 2020.
4. Run `python setup.py build_ext`. This will build and compile the module
5. Run `python setup.py install`. This will install the package as `FL` in your local python installation. If you are using a virtual environment (recommended) the package will only be installed in your virtual env.
6. Use the package as shown in the Usage section or see the examples in `demo/`

# Setup Windows (Only works with Windows right now. I'll update environment later.):

1. Install Microsoft MPI v10.0 from https://www.microsoft.com/en-us/download/details.aspx?id=57467&WT.mc_id=rss_alldownloads_devresources. Make sure to add C:\Program Files\Microsoft MPI\Bin\ to your "Environment Variables".

2. This project uses Anaconda environments to guarantee package management. There is a .yml file provided. Open the command prompt. Please build a conda environment from this file using the command "conda env create -n FLRR -f windows_environment.yml". Activate the environment using "conda activate FLRR". Verify that the new environment is working using conda env list.

# Setup using VirtualEnv

1. Create a virtual env using `python3 -m venv .venv` and then use it by running `source .venv/bin/activate`
2. `pip install -r requirements.txt` will install all needed packages. If you update the source code and add a new package, make sure to update the requirements file by running `pip freeze > requirements.txt`

# Usage (Legacy):

1. With the conda environment activated use the command "mpiexec -n 5 python fedavg_demo.py"

2. 5 can be replaced with any number of processes. Performance will be based on the number of cores of the computer. At the end of the simulation, a picture of the MSE performance will be generated. Note that process "0" or as in the code "my_rank = 0" denotes the server. Hence, when we use 5 processes we will have 1 server and 4 workers.

3. If one wants to run a different simulation, copy the fedavg_demo.py file.

4. Open the .py file and see the different parameters that can be modified such as the number of experiments, dimension of a data point, etc.

5. Currently usage is limited to l2-regularized logisitic regression. We can extend the functionality to different types of problem by creating a child-class from the base class Classifier and providing an objective/gradient function. See the details in the classifier.py file.
