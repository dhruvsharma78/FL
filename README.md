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

1. Install required packages listed in requirements.txt. It is advised to use a virtual environment here. See later section on how to do this. Some environments may also require that you install cmake and cpython.
2. Install OpenMP, OpenMPI and Intel MKL. While Intel oneMKL may work. This version has only been tested with MKL 2020.
3. Ensure that your shell has the MKL environment variables. To load these, you can source the `mklvars.sh` file provided in the MKL installation.
4. In setup.py change the global variables `extra_compile_args`, `extra_link_args`, `lib_paths` and `libs` to match your installation of MKL and OpenMP along with the compiler you wish to use. Use the [Intel MKL Link Line Advisor](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html) to find your link and compile arguments. The default setup works with `gcc` and uses the TBB threaded version of Intel MKL 2020. We recommended you use gcc. Make sure your gcc version was compiled with OpenMP support.
5. Run `python setup.py build_ext`. This will build and compile the module
6. Run `python setup.py install`. This will install the package as `FL` in your local python installation. If you are using a virtual environment (recommended) the package will only be installed in your virtual env.
7. Use the package as shown in the Usage section or see the examples in `demo/`

# Setup using VirtualEnv

1. Create a virtual env using `python3 -m venv .venv` and then use it by running `source .venv/bin/activate`
2. `pip install -r requirements.txt` will install all needed packages. If you update the source code and add a new package, make sure to update the requirements file by running `pip freeze > requirements.txt`

# Usage (Legacy):

1. With the conda environment activated use the command "mpiexec -n 5 python fedavg_demo.py"

2. 5 can be replaced with any number of processes. Performance will be based on the number of cores of the computer. At the end of the simulation, a picture of the MSE performance will be generated. Note that process "0" or as in the code "my_rank = 0" denotes the server. Hence, when we use 5 processes we will have 1 server and 4 workers.

3. If one wants to run a different simulation, copy the fedavg_demo.py file.

4. Open the .py file and see the different parameters that can be modified such as the number of experiments, dimension of a data point, etc.

5. Currently usage is limited to l2-regularized logisitic regression. We can extend the functionality to different types of problem by creating a child-class from the base class Classifier and providing an objective/gradient function. See the details in the classifier.py file.
