## LEAF-GP Benchmarks

## Installation - Linux

### Creating a virtual environment

We use `virtualenv` to setup a virtual environment.
You can install this package by running:
```
python3 -m pip install virtualenv
```
To set up a new virtual environment called 'env' with Python 3.7 for which this code was tested, run the command:
```
python3 -m virtualenv env --python=python3.7
```
in the folder where you want to store the virtual environment.
Afterwards, activate the environment using
```
source env/bin/activate
```
It is recommended that you update the pip installation in the virtual environment:
```
pip install --upgrade pip
```
Install all required packages by running the command:
```
pip install -r requirements.txt
```

### Installlation of Gurobi 9
Please visit the Gurobi [website](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) to 
receive an academic license and download the solver.
To install the optimization modelling environment run:
```
python -m pip install -i https://pypi.gurobi.com gurobipy
```

## Run black-box function benchmarks
As stated in the paper we evaluate black-box functions: `hartmann6d`, `rastrigin`, `styblinski_tang`, `schwefel`, 
`g1`, `g3`, `g4`, `g6`, `g7`, `g10`, `alkylation`, `pressure_vessel` and `vae_nas`.
To test `LEAF-GP` with the `hartmann6d` benchmark function run:
```
python run_study.py -bb-func hartmann6d
```
You can also modify the call by using optional arguments:
- `-num-init`: number of initial data points
- `-num-itr`: number of optimization iterations
- `-rnd-seed`: random seed to evaluate
- `-solver-type`: pick either `global` or `sampling`, referring to `LEAF-GP` and `LEAF-GP-RND`, respectively
- `-has-larger-model`: picking this one uses a larger tree ensemble model for `LEAF-GP` 
  used for the `vae_nas` benchmark