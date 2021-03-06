import numpy as np
from leaf_gp.leaf_gp_utils import LeafGP

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-bb-func", type=str, default="hartmann6d")
parser.add_argument("-num-init", type=int, default=5)
parser.add_argument("-num-itr", type=int, default=100)
parser.add_argument("-rnd-seed", type=int, default=101)
parser.add_argument("-solver-type", type=str, default="global") # can also be 'sampling'
parser.add_argument("-has-larger-model", action='store_true')
args = parser.parse_args()

# set random seeds for reproducibility
from leaf_gp.helper import set_rnd_states
set_rnd_states(args)

# load black-box function to evaluate
from bb_func_utils import get_func
bb_func = get_func(args.bb_func)

# activate label encoding if categorical features are given
if bb_func.cat_idx:
    bb_func.eval_label()

# generate initial data points
init_data = bb_func.get_init_data(args.num_init, args.rnd_seed)
X, y = init_data['X'], init_data['y']

print(f"* * * initial data targets:")
for yi in y:
    print(f"  val: {round(yi, 4)}")

# add model_core with constraints if problem has constraints
if bb_func.has_constr():
    model_core = bb_func.get_model_core()
else:
    model_core = None

# modify tree model hyperparameters
if not args.has_larger_model:
    tree_params = {'boosting_rounds': 50,
                   'max_depth': 3,
                   'min_data_in_leaf': 1}
else:
    tree_params = {'boosting_rounds': 100,
                   'max_depth': 5,
                   'min_data_in_leaf': 1}

# main bo loop
print(f"\n* * * start bo loop...")
for itr in range(args.num_itr):

    # prepare and train model
    model = LeafGP(bb_func.get_space(),
                   kappa=1.96,
                   model_core=model_core,
                   random_state=args.rnd_seed,
                   tree_params=tree_params,
                   solver_type=args.solver_type)

    X_train, y_train = np.asarray(X), np.asarray(y)

    model.fit(X_train, y_train)

    # get new proposal and evaluate bb_func
    next_x = model.propose()
    next_y = bb_func(next_x)

    # update progress
    X.append(next_x)
    y.append(next_y)

    print(f"{itr}. min_val: {round(min(y), 5)}")