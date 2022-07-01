import numpy as np
import random

# gp imports
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

class LeafGP:

    def __init__(self,
                 space,
                 kappa=1.96,
                 model_core=None,
                 random_state=None,
                 tree_params=None,
                 gp_params=None,
                 solver_type='global'):

        # define attributes from input parameters
        self.space = space
        self.lb = np.asarray([dim.bnds[0] for dim in self.space.dims])
        self.ub = np.asarray([dim.bnds[1] for dim in self.space.dims])
        self.X, self.y = None, None

        # kappa defines exploitation - exploration trade-off
        self.kappa = kappa
        self.model_core = model_core

        self.random_state = random_state

        self.solver_type = solver_type

        # set random state
        if self.random_state:
            random.seed(self.random_state)

        # define tree ensemble and gp hyperparameters
        self.tree_params = {} if tree_params is None else tree_params
        self.gp_params = {} if gp_params is None else gp_params

        # get kernel hyperparameters
        if 'tree_kernel' not in self.gp_params:
            self.gp_params['tree_kernel'] = {}

        # for tree gp
        from gpytorch.constraints.constraints import Interval
        self.noise_constraint = \
            self.gp_params['tree_kernel'].get('noise_constraint', Interval(5e-4, 0.2))
        self.outputscale_constraint = \
            self.gp_params['tree_kernel'].get('outputscale_constraint', Interval(0.05, 20.0))
        self.epochs = self.gp_params.get('epochs', 200)

        self.tree_gp_model = None
        self.tree_model = None

    def fit(self, X, y):
        self.X, self.y = X, y

        ## train tree ensemble in lgbm
        boosting_rounds = self.tree_params.get('boosting_rounds', 50)
        max_depth = self.tree_params.get('max_depth', 3)
        min_data_in_leaf = self.tree_params.get('min_data_in_leaf', 1)

        from leaf_gp.tree_training import WrapperLGBM
        tree_model = WrapperLGBM(self.space)
        tree_model.fit(X, y,
                       boosting_rounds=boosting_rounds,
                       max_depth=max_depth,
                       min_data_in_leaf=min_data_in_leaf)

        ## train tree GP
        import torch

        # convert data set and standardize it
        train_x = torch.from_numpy(X).double()
        train_y = torch.from_numpy(y).double()

        from botorch.utils import standardize
        train_y = standardize(train_y)

        # define gaussian likelihood
        from gpytorch.likelihoods import GaussianLikelihood
        tree_likelihood = \
            GaussianLikelihood(noise_constraint=self.noise_constraint)

        # define tree kernel
        from gpytorch.kernels import ScaleKernel
        tree_kernel = tree_model.get_kernel()
        full_kernel = ScaleKernel(tree_kernel,
                                  outputscale_constraint=self.outputscale_constraint)

        tree_gp_model = GPModel(train_x,
                                train_y,
                                tree_likelihood,
                                full_kernel)

        self._train_gp(tree_gp_model, train_x, train_y,
                       epochs=self.epochs)

        # define all trained models
        self.tree_gp_model = tree_gp_model
        self.tree_model = tree_model
        self.tree_kernel = tree_kernel

    def _train_gp(self,
                  model,
                  train_x, train_y,
                  epochs=200):

        ## train the gp using separat adam solver
        from gpytorch.mlls import ExactMarginalLogLikelihood
        import torch

        # use the adam optimizer
        optimizer = \
            torch.optim.Adam(model.parameters(),
                             lr=0.1)  # Includes GaussianLikelihood parameters

        model.double()
        model.train()
        model.likelihood.train()

        # "loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # suppresses gpytorch depreciation warning for better readability
        import warnings
        with warnings.catch_warnings(record=True) as w:
            for _ in range(epochs):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

    def _build_model(self):
        # build opt_model core
        from leaf_gp.optimizer_utils import \
            get_opt_core, add_gbm_to_opt_model, get_opt_core_copy

        # check if there's already a model core with extra constraints
        if self.model_core is None:
            opt_model = get_opt_core(self.space)
        else:
            # copy model core in case there are constr given already
            opt_model = get_opt_core_copy(self.model_core)

        # build tree model
        assert self.tree_gp_model is not None, \
            f"fit a model first before proposing new points"
        gbm_model_dict = {'1st_obj': self.tree_model._gbm_model}
        add_gbm_to_opt_model(self.space,
                             gbm_model_dict,
                             opt_model,
                             z_as_bin=True)

        # get tree_gp hyperparamters
        kernel_var = self.tree_gp_model.covar_module.outputscale.detach().numpy()
        noise_var = self.tree_gp_model.likelihood.noise.detach().numpy()

        # get tree_gp matrices
        import torch
        train_x = torch.from_numpy(self.X).double()
        Kmm = self.tree_gp_model.covar_module(train_x).numpy()
        k_diag = np.diagonal(Kmm)
        s_diag = self.tree_gp_model.likelihood._shaped_noise_covar(k_diag.shape).numpy()
        ks = Kmm + s_diag

        # invert gram matrix
        from scipy.linalg import cho_solve, cho_factor
        id_ks = np.eye(ks.shape[0])
        inv_ks = cho_solve(
            cho_factor(ks, lower=True), id_ks)

        # add tree_gp logic to opt_model
        from gurobipy import GRB, MVar

        act_leave_vars = self.tree_model.get_active_leaf_vars(
            self.X,
            opt_model,
            '1st_obj')

        sub_k = opt_model.addVars(
            range(len(act_leave_vars)),
            lb=0,
            ub=1,
            name="sub_k",
            vtype='C')

        opt_model.addConstrs(
            (sub_k[idx] == act_leave_vars[idx]
             for idx in range(len(act_leave_vars))),
            name='sub_k_constr')

        ## add quadratic constraints
        opt_model._var = opt_model.addVar(
            lb=0,
            ub=GRB.INFINITY,
            name="var",
            vtype='C')

        opt_model._sub_k_var = MVar(
            [sub_k[id]
             for id in range(len(sub_k))] + [opt_model._var])

        quadr_term = - (kernel_var ** 2) * inv_ks
        const_term = kernel_var + noise_var

        quadr_constr = np.zeros(
            (quadr_term.shape[0] + 1,
            quadr_term.shape[1] + 1))
        quadr_constr[:-1, :-1] = quadr_term
        quadr_constr[-1, -1] = -1.0

        opt_model.addMQConstr(
            quadr_constr, None,
            sense='>',
            rhs=-const_term,
            xQ_L=opt_model._sub_k_var,
            xQ_R=opt_model._sub_k_var)

        ## add linear objective
        opt_model._sub_z_obj = MVar(
            [sub_k[idx] for idx in range(len(sub_k))] + [opt_model._var])

        y_vals = self.tree_gp_model.train_targets.numpy().tolist()
        lin_term = kernel_var * np.matmul(
            inv_ks, np.asarray(y_vals))

        lin_obj = np.zeros(len(lin_term) + 1)
        lin_obj[:-1] = lin_term
        lin_obj[-1] = - self.kappa

        opt_model.setMObjective(
            None, lin_obj, 0,
            xc=opt_model._sub_z_obj,
            sense=GRB.MINIMIZE)

        ## add mu variable
        opt_model._sub_z_mu = MVar(
            [sub_k[idx] for idx in range(len(sub_k))])
        opt_model._mu_coeff = lin_term

        return opt_model

    def _get_sampling_sol(self, num_samples=2000):
        # generates a solution based on sampling the acquisition function
        from leaf_gp.optimizer_utils import conv2list
        from skopt.space.space import Space as SkoptSpace
        from skopt.space.space import Categorical, Integer, Real

        skopt_bnds = []

        for idx, d in enumerate(self.space.dims):
            if idx in self.space.cat_idx:
                skopt_bnds.append(Categorical(d.bnds, transform='label'))
            elif idx in self.space.int_idx:
                skopt_bnds.append(Integer(low=int(d.bnds[0]), high=int(d.bnds[1])))
            else:
                skopt_bnds.append(Real(low=float(d.bnds[0]), high=float(d.bnds[1])))

        skopt_space = SkoptSpace(skopt_bnds)
        samples = skopt_space.rvs(num_samples, random_state=self.random_state)

        mean, std = self.predict(samples, return_std=True)
        acq = mean - self.kappa * std
        min_idx = np.argmin(acq)

        # get best mean and std combination
        curr_mean = mean[min_idx]
        curr_std = std[min_idx]

        # get solution
        next_x = conv2list(samples[min_idx])

        # get active area
        var_bnds = [d.bnds for d in self.space.dims]
        active_enc = self.tree_model._gbm_model.get_active_leaves(next_x)
        active_enc_tuple = [(tree_id, enc) for tree_id, enc in enumerate(active_enc)]
        self.tree_model._gbm_model.update_var_bounds(active_enc_tuple, var_bnds)

        return var_bnds, next_x, curr_mean, curr_std

    def _get_global_sol(self):
        # provides global solution to the optimization problem

        # build main model
        opt_model = self._build_model()

        ## set solver parameters
        opt_model.Params.LogToConsole = 0
        opt_model.Params.Heuristics = 0.2
        opt_model.Params.TimeLimit = 100

        ## optimize opt_model to determine area to focus on
        opt_model.optimize()

        # get active leaf area
        from leaf_gp.optimizer_utils import label_leaf_index
        label = '1st_obj'
        var_bnds = [d.bnds for d in self.space.dims]

        active_enc = \
            [(tree_id, leaf_enc) for tree_id, leaf_enc in label_leaf_index(opt_model, label)
            if round(opt_model._z_l[label, tree_id, leaf_enc].x) == 1.0]
        self.tree_model._gbm_model.update_var_bounds(active_enc, var_bnds)

        # reading x_val
        from leaf_gp.optimizer_utils import get_opt_sol
        next_x = get_opt_sol(self.space, opt_model)

        # extract variance and mean
        curr_var = opt_model._var.x
        curr_mean = sum([opt_model._mu_coeff[idx]*opt_model._sub_z_mu[idx].x
                         for idx in range(len(opt_model._mu_coeff))])

        return var_bnds, next_x, curr_mean, curr_var

    def propose_leaf(self):
        if self.solver_type == 'sampling':
            num_samples = 2000
            return self._get_sampling_sol(num_samples=num_samples)

        elif self.solver_type == 'global':
            return self._get_global_sol()
        else:
            raise ValueError(f"solver_type '{self.solver_type}' is not supported!")

    def _add_epsilon_to_bnds(self, x_area):
        # adds a 1e-5 error to the bounds of area
        eps = 10**(-5)
        for idx in range(len(self.space.dims)):
            if idx not in self.space.cat_idx:
                lb, ub = x_area[idx]
                new_lb = max(lb - eps, self.space.dims[idx].bnds[0])
                new_ub = min(ub + eps, self.space.dims[idx].bnds[1])
                x_area[idx] = (new_lb, new_ub)

    def _get_leaf_center(self, x_area):
        """returns the center of x_area"""
        next_x = []
        for idx in range(len(x_area)):
            if idx in self.space.cat_idx:
                # for cat vars
                xi = int(np.random.choice(list(x_area[idx]), size=1)[0])
            else:
                lb, ub = x_area[idx]

                if self.space.dims[idx].is_bin:
                    # for bin vars
                    if lb == 0 and ub == 1:
                        xi = int(np.random.randint(0, 2))
                    elif lb <= 0.1:
                        xi = 0
                    elif ub >= 0.9:
                        xi = 1
                    else:
                        raise ValueError("problem with binary split, go to 'get_leaf_center'")

                elif idx in self.space.int_idx:
                    # for int vars
                    lb, ub = round(lb), round(ub)
                    m = lb + (ub - lb) / 2
                    xi = int(np.random.choice([int(m), round(m)], size=1)[0])

                else:
                    # for conti vars
                    xi = float(lb + (ub - lb) / 2)

            next_x.append(xi)
        return next_x

    def _get_leaf_min_center_dist(self, x_area):
        """returns the feasible point closest to the x_area center"""
        # build opt_model core
        from leaf_gp.optimizer_utils import \
            get_opt_core, get_opt_core_copy

        # check if there's already a model core with extra constraints
        if self.model_core is None:
            opt_model = get_opt_core(self.space)
        else:
            # copy model core in case there are constr given already
            opt_model = get_opt_core_copy(self.model_core)


        # define alpha as the distance to closest data point
        from gurobipy import GRB
        opt_model._alpha = opt_model.addVar(lb=0.0, ub=GRB.INFINITY, name='alpha')

        # update bounds for all variables
        for idx in range(len(self.space.dims)):
            if idx in self.space.cat_idx:
                # add constr for cat vars
                cat_set = set(self.space.dims[idx].bnds)

                for cat in cat_set:
                    # cat is fixed to what is valid with respect to x_area[idx]
                    if cat not in x_area[idx]:
                        opt_model.addConstr(opt_model._cat_var_dict[idx][cat] == 0)
            else:
                lb, ub = x_area[idx]
                opt_model.addConstr(opt_model._cont_var_dict[idx] <= ub)
                opt_model.addConstr(opt_model._cont_var_dict[idx] >= lb)

        # add constraints for every data point
        x_center = self._get_leaf_center(x_area)

        for x in [x_center]:
            expr = []

            # add dist for all dimensions
            for idx in range(len(self.space.dims)):
                if idx in self.space.cat_idx:
                    # add constr for cat vars
                    cat_set = set(self.space.dims[idx].bnds)

                    for cat in cat_set:
                        # distance increases by one if cat is different from x[idx]
                        if cat != x[idx]:
                            expr.append(opt_model._cat_var_dict[idx][cat])

                else:
                    # add constr for conti and int vars
                    expr.append((x[idx] - opt_model._cont_var_dict[idx])**2)

            # add dist constraints to model
            opt_model.addConstr(opt_model._alpha >= sum(expr))

        # set optimization parameters
        opt_model.Params.LogToConsole = 0
        opt_model.Params.NonConvex = 2
        opt_model.setObjective(expr=opt_model._alpha)
        opt_model.optimize()

        from leaf_gp.optimizer_utils import get_opt_sol
        return get_opt_sol(self.space, opt_model)

    def propose(self):

        next_x_area, next_val, curr_mean, curr_var = \
            self.propose_leaf()

        # add epsilon if input constr. exist
        # i.e. tree splits are rounded to the 5th decimal when adding them to the model,
        # and this may make optimization problems infeasible if the feasible region is very small
        if self.model_core:
            self._add_epsilon_to_bnds(next_x_area)

            while True:
                try:
                    next_center = self._get_leaf_min_center_dist(next_x_area)
                    break
                except RuntimeError:
                    self._add_epsilon_to_bnds(next_x_area)
        else:
            next_center = self._get_leaf_center(next_x_area)

        return next_center

    def predict(self, X, return_std=False):
        import torch
        X = np.asarray(X)

        eval_x = torch.from_numpy(X).double()
        self.tree_gp_model.eval()

        with torch.no_grad():
            pred = self.tree_gp_model(eval_x)
            pred_lik = self.tree_gp_model.likelihood(pred)

            mean = pred_lik.mean.numpy()

            if return_std:
                std = np.sqrt(pred_lik.variance.numpy())
                return mean, std

            return mean

class GPModel(ExactGP):
    def __init__(self,
                 train_x, train_y,
                 likelihood,
                 kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
