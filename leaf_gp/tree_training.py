import numpy as np

class WrapperLGBM:
    def __init__(self, space):
        self.space = space
        self.estimator = None

    def _train_gbrt(self, X_train, y_train, boosting_rounds, params, cat_idx=None):
        # train lgbm tree ensemble
        import lightgbm as lgb

        if cat_idx:
            params["min_data_per_group"] = 1
            train_data = lgb.Dataset(X_train, label=y_train,
                                     categorical_feature=cat_idx,
                                     params={'verbose': -1})
            model = lgb.train(params,
                              train_data,
                              categorical_feature=cat_idx,
                              num_boost_round=boosting_rounds)
        else:
            train_data = lgb.Dataset(X_train, label=y_train,
                                     params={'verbose': -1})
            model = lgb.train(params,
                              train_data,
                              num_boost_round=boosting_rounds)
        return model

    def fit(self,
            X_train,
            y_train,
            boosting_rounds=50,
            max_depth=3,
            min_data_in_leaf=1):

        # modify parameters for tree ensemble
        tree_params = {'boosting_type': 'gbdt',
                       'max_depth': max_depth,
                       'min_data_in_leaf': min_data_in_leaf,
                       'verbose': -1}

        model = self._train_gbrt(X_train, y_train,
                                 boosting_rounds,
                                 tree_params,
                                 cat_idx=self.space.cat_idx)

        self.estimator = model

    def predict(self, X):
        """predicts locations X using the fitted ensemble"""
        if self.estimator is None:
            raise RuntimeError("fit the model first before making predictions")
        else:
            return self.estimators.predict(X)

    def get_kernel(self):
        if self.estimator is None:
            raise RuntimeError("fit the model first before exporting the kernel")

        # dump tree ensemble and format as python object
        from leaf_gp.lgbm_processing import order_tree_model_dict
        from leaf_gp.gbm_model import GbmModel

        original_tree_model_dict = self.estimator.dump_model()

        ordered_tree_model_dict = \
            order_tree_model_dict(original_tree_model_dict,
                                  cat_column=self.space.cat_idx)

        self._gbm_model = GbmModel(ordered_tree_model_dict)

        # build the tree kernel
        from gpytorch.kernels import Kernel
        import torch

        class TreeKernel(Kernel):
            is_stationary = False

            def __init__(self, gbm_model):
                super().__init__()
                self._gbm_model = gbm_model

                # attributes introduced for leaf optimization
                self._leaf_eval = False
                self._kernel_cache = None
                self._leaf_center = None
                self._var_bnds = None
                self._input_trafo = None

            def forward(self, x1, x2, diag=False, **params):
                # normal evaluation without any modifications
                np_x1 = x1.detach().numpy()
                np_x2 = x2.detach().numpy()

                out = self._gbm_model.get_gram_mat(np_x1, np_x2)

                if diag:
                    out = torch.ones(x1.shape[0])

                return torch.as_tensor(out, dtype=x1.dtype)

            def get_tree_agreement(self, x1, x2):
                return self._gbm_model.get_gram_mat(x1, x2)

        return TreeKernel(self._gbm_model)

    def get_active_leaf_vars(self, X, model, gbm_label):
        # get active leaves for X
        act_leaves_x = np.asarray(
            [self._gbm_model.get_active_leaves(x) for x in X]
        )

        # generate active_leave_vars
        act_leaf_vars = []
        for data_id, data_enc in enumerate(act_leaves_x):
            temp_lhs = 0
            for tree_id, leaf_enc in enumerate(data_enc):
                temp_lhs += model._z_l[gbm_label, tree_id, leaf_enc]

            temp_lhs *= 1 / len(data_enc)

            act_leaf_vars.append(temp_lhs)

        act_leaf_vars = np.asarray(act_leaf_vars)
        return act_leaf_vars
