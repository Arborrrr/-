from data_preprocessing import *


def tune_ElasticNet(trial):  # 优化ElasticNet的超参数
    max_iter = trial.suggest_int("max_iter", 1000, 4000)
    alpha = trial.suggest_float("alpha", 1e-4, 1000, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.05)
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    ElasticNet_regressor = ElasticNet(max_iter=max_iter, alpha=alpha, tol=tol, l1_ratio=l1_ratio, random_state=1)

    # -- Make a pipeline
    ElasticNet_pipeline = make_pipeline(linear_preprocessor, ElasticNet_regressor)

    ElasticNet_model = TransformedTargetRegressor(regressor=ElasticNet_pipeline, func=np.log1p, inverse_func=np.expm1)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(ElasticNet_model, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_xgboost_1(trial):  # 第一级xgboost优化
    n_estimators1 = trial.suggest_int("n_estimators", 2000, 5000, step=500)

    # L1 regularization weight
    alpha1 = trial.suggest_float("alpha", 1e-8, 1.0, log=True)

    # sampling ratio for training data
    subsample1 = trial.suggest_float("subsample", 0.2, 1.0)

    # sampling according to each tree
    colsample_bytree1 = trial.suggest_float("colsample_bytree", 0.4, 0.6)

    # maximum depth of the tree, signifies complexity of the tree
    max_depth1 = trial.suggest_int("max_depth", 3, 10, step=2)

    # minimum child weight, larger the term more conservative the tree
    min_child_weight1 = trial.suggest_int("min_child_weight", 1, 3)

    # learning rate
    learning_rate1 = trial.suggest_float("learning_rate", 1e-6, 1, log=True)

    # defines how selective algorithm is
    gamma1 = trial.suggest_float("gamma", 1e-8, 1.0, log=True)

    xgb_regressor = XGBRegressor(
        n_estimators=n_estimators1,
        alpha=alpha1,
        subsample=subsample1,
        colsample_bytree=colsample_bytree1,
        max_depth=max_depth1,
        min_child_weight=min_child_weight1,
        learning_rate=learning_rate1,
        gamma=gamma1,
        eval_metric='mae',
        random_state=1
    )

    # -- Make a pipeline
    xgb_pipeline = make_pipeline(tree_preprocessor, xgb_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(xgb_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_xgboost_2(trial):  # 第二级xgboost优化
    n_estimators1 = trial.suggest_int("n_estimators", 4000, 6000, step=500)

    # L1 regularization weight
    alpha1 = trial.suggest_float("alpha", 1e-8, 1e-5, log=True)

    # sampling ratio for training data
    subsample1 = trial.suggest_float("subsample", 0.3, 0.5)

    # sampling according to each tree
    colsample_bytree1 = trial.suggest_float("colsample_bytree", 0.35, 0.56)

    # maximum depth of the tree, signifies complexity of the tree
    max_depth1 = trial.suggest_int("max_depth", 3, 7, step=2)

    # minimum child weight, larger the term more conservative the tree
    min_child_weight1 = trial.suggest_int("min_child_weight", 1, 3)

    # learning rate
    learning_rate1 = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # defines how selective algorithm is
    gamma1 = trial.suggest_float("gamma", 1e-3, 1e-1, log=True)

    xgb_regressor = XGBRegressor(
        n_estimators=n_estimators1,
        alpha=alpha1,
        subsample=subsample1,
        colsample_bytree=colsample_bytree1,
        max_depth=max_depth1,
        min_child_weight=min_child_weight1,
        learning_rate=learning_rate1,
        gamma=gamma1,
        eval_metric='mae',
        random_state=1
    )

    # -- Make a pipeline
    xgb_pipeline = make_pipeline(tree_preprocessor, xgb_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(xgb_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_xgboost_3(trial):  # 第三级xgboost优化
    n_estimators1 = trial.suggest_int("n_estimators", 6000, 7000, step=500)

    # L1 regularization weight
    alpha1 = trial.suggest_float("alpha", 1e-9, 1e-7, log=True)

    # sampling ratio for training data
    subsample1 = trial.suggest_float("subsample", 0.3, 0.4)

    # sampling according to each tree
    colsample_bytree1 = trial.suggest_float("colsample_bytree", 0.25, 0.45)

    # maximum depth of the tree, signifies complexity of the tree
    max_depth1 = trial.suggest_int("max_depth", 3, 7, step=2)

    # minimum child weight, larger the term more conservative the tree
    min_child_weight1 = trial.suggest_int("min_child_weight", 1, 3)

    # learning rate
    learning_rate1 = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)

    # defines how selective algorithm is
    gamma1 = trial.suggest_float("gamma", 1e-3, 1e-1, log=True)

    xgb_regressor = XGBRegressor(
        n_estimators=n_estimators1,
        alpha=alpha1,
        subsample=subsample1,
        colsample_bytree=colsample_bytree1,
        max_depth=max_depth1,
        min_child_weight=min_child_weight1,
        learning_rate=learning_rate1,
        gamma=gamma1,
        eval_metric='mae',
        random_state=1
    )

    # -- Make a pipeline
    xgb_pipeline = make_pipeline(tree_preprocessor, xgb_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(xgb_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_gbm_1(trial):  # 第一级gbm优化
    n_estimators = trial.suggest_int("n_estimators", 5000, 6000, step=500)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 22, step=2)
    max_depth = trial.suggest_int("max_depth", 2, 7, step=1)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    subsample = trial.suggest_float("subsample", 0.85, 0.95)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    gbm_regressor = GradientBoostingRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        max_features=max_features,
        random_state=1
    )

    # -- Make a pipeline
    gbm_pipeline = make_pipeline(tree_preprocessor, gbm_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(gbm_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_gbm_2(trial):  # 第二级gbm优化
    n_estimators = trial.suggest_int("n_estimators", 5000, 6000, step=500)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 22, step=2)
    max_depth = trial.suggest_int("max_depth", 2, 7, step=1)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    subsample = trial.suggest_float("subsample", 0.85, 0.95)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    gbm_regressor = GradientBoostingRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        max_features=max_features,
        random_state=1
    )

    # -- Make a pipeline
    gbm_pipeline = make_pipeline(tree_preprocessor, gbm_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(gbm_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_catboost(trial):  # 优化catboost的超参数
    cat_param = {
        "iterations": trial.suggest_int("iterations", 4000, 6500, step=500),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.02, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 1, log=True),
        "depth": trial.suggest_int("depth", 2, 12),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 8),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli"])
    }

    if cat_param["bootstrap_type"] == "Bayesian":
        cat_param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif cat_param["bootstrap_type"] == "Bernoulli":
        cat_param["subsample"] = trial.suggest_float("subsample", 0.6, 1)

    catboost_regressor = CatBoostRegressor(**cat_param, random_state=1, logging_level='Silent')

    # -- Make a pipeline
    catboost_pipeline = make_pipeline(tree_preprocessor, catboost_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(catboost_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_lightgbm_1(trial):  # 第一级lightgbm优化
    n_estimators = trial.suggest_int("n_estimators", 1500, 5000, step=500)
    max_depth = trial.suggest_int("max_depth", 2, 14, step=2)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 40, step=3)
    subsample = trial.suggest_float("subsample", 0.6, 1.0, step=0.05)
    max_bin = trial.suggest_int("max_bin", 200, 350, step=10),
    feature_fraction = trial.suggest_float("feature_fraction", 0.3, 1.0, step=0.1)

    lgbm_regressor = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        subsample=subsample,
        max_bin=max_bin,
        feature_fraction=feature_fraction,
        random_state=1
    )

    # -- Make a pipeline
    lgbm_pipeline = make_pipeline(tree_preprocessor, lgbm_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(lgbm_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_lightgbm_2(trial):  # 第二级lightgbm优化
    n_estimators = trial.suggest_int("n_estimators", 5500, 7000, step=500)
    max_depth = trial.suggest_int("max_depth", 4, 12)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 21, 30)
    subsample = trial.suggest_float("subsample", 0.6, 0.88)
    max_bin = trial.suggest_int("max_bin", 190, 230, step=10),
    feature_fraction = trial.suggest_float("feature_fraction", 0.3, 0.5)

    lgbm_regressor = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        subsample=subsample,
        max_bin=max_bin,
        feature_fraction=feature_fraction,
        random_state=1
    )

    # -- Make a pipeline
    lgbm_pipeline = make_pipeline(tree_preprocessor, lgbm_regressor)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(lgbm_pipeline, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_Lasso(trial):  # 优化Lasso的超参数
    max_iter = trial.suggest_int("max_iter", 1000, 4000)
    alpha = trial.suggest_float("alpha", 1e-4, 1000, log=True)
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    Lasso_regressor = Lasso(
        max_iter=max_iter,
        alpha=alpha,
        tol=tol,
        random_state=1
    )

    # -- Make a pipeline
    Lasso_pipeline = make_pipeline(linear_preprocessor, Lasso_regressor)

    Lasso_model = TransformedTargetRegressor(regressor=Lasso_pipeline, func=np.log1p, inverse_func=np.expm1)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(Lasso_model, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_Ridge(trial):  # 优化Ridge的超参数
    max_iter = trial.suggest_int("max_iter", 1000, 4000)
    alpha = trial.suggest_float("alpha", 1e-4, 1000, log=True)
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    Ridge_regressor = Ridge(
        max_iter=max_iter,
        alpha=alpha,
        tol=tol,
        random_state=1
    )

    # -- Make a pipeline
    Ridge_pipeline = make_pipeline(linear_preprocessor, Ridge_regressor)

    Ridge_model = TransformedTargetRegressor(regressor=Ridge_pipeline, func=np.log1p, inverse_func=np.expm1)

    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(Ridge_model, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


# Be careful higher C values takes too long or fails to converge
def tune_SVR_1(trial):  # 第一级SVR优化
    param = {
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        "C": trial.suggest_float("C", 1, 1000, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 0.1, log=True),
        "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True)
    }

    if param["kernel"] == "rbf":
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    SVR_regressor = SVR(
        **param,
        cache_size=800
    )

    # -- Make a pipeline
    SVR_pipeline = make_pipeline(linear_preprocessor, SVR_regressor)

    SVR_model = TransformedTargetRegressor(regressor=SVR_pipeline, func=np.log1p, inverse_func=np.expm1)
    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(SVR_model, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


def tune_SVR_2(trial):  # 第二级SVR优化
    param = {
        "kernel": trial.suggest_categorical("kernel", ["linear"]),
        "C": trial.suggest_float("C", 1e-3, 10, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-4, 0.1, log=True),
        "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True)
    }

    SVR_regressor = SVR(
        **param,
        cache_size=800
    )

    # -- Make a pipeline
    SVR_pipeline = make_pipeline(linear_preprocessor, SVR_regressor)

    SVR_model = TransformedTargetRegressor(regressor=SVR_pipeline, func=np.log1p, inverse_func=np.expm1)
    ss = ShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0
    )  # ShuffleSplit will help you to keep distributions of training data and folds similar)

    score = cross_val_score(SVR_model, pipe_data, y, scoring=make_scorer(mean_absolute_error), cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)  # create a seed for the sampler for reproducibility

# 优化ElasticNet
study_ElasticNet = optuna.create_study(direction="minimize", sampler=sampler)
study_ElasticNet.optimize(tune_ElasticNet, n_trials=100)

# 优化XGBRegressor
study_xgboost = optuna.create_study(direction="minimize", sampler=sampler)
study_xgboost.optimize(tune_xgboost_1, n_trials=100)
study_xgboost.optimize(tune_xgboost_2, n_trials=100)
study_xgboost.optimize(tune_xgboost_3, n_trials=100)

# 优化GradientBoostingRegressor
study_gbm = optuna.create_study(direction="minimize", sampler=sampler)
study_gbm.optimize(tune_gbm_1, n_trials=100)
study_gbm.optimize(tune_gbm_2, n_trials=100)

# 优化catboost
study_catboost = optuna.create_study(direction="minimize", sampler=sampler)
study_catboost.optimize(tune_catboost, n_trials=100)

# 优化lightgbm
study_lightgbm = optuna.create_study(direction="minimize", sampler=sampler)
study_lightgbm.optimize(tune_lightgbm_1, n_trials=100)
study_lightgbm.optimize(tune_lightgbm_2, n_trials=100)

# 优化Lasso
study_Lasso = optuna.create_study(direction="minimize", sampler=sampler)
study_Lasso.optimize(tune_Lasso, n_trials=100)

# 优化Ridge
study_Ridge = optuna.create_study(direction="minimize", sampler=sampler)
study_Ridge.optimize(tune_Ridge, n_trials=100)

# 优化SVR
study_SVR = optuna.create_study(direction="minimize", sampler=sampler)
study_SVR.optimize(tune_SVR_1, n_trials=100)
study_SVR.optimize(tune_SVR_2, n_trials=100)
