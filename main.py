from data_preprocessing import *

xgb_tunned = XGBRegressor(
    n_estimators=6500,
    alpha=1.7938525031017074e-09,
    subsample=0.3231512729662032,
    colsample_bytree=0.25528017285233484,
    max_depth=5,
    min_child_weight=2,
    learning_rate=0.004828231865923587,
    gamma=0.0026151163125498213,
    random_state=1
)

pipe_xgb = Pipeline(
    steps=[
        ('tree_preprocessor', tree_preprocessor),
        ('regressor1', xgb_tunned)
    ]
)

gbm_tunned = GradientBoostingRegressor(
    n_estimators=5500,
    max_depth=5,
    min_samples_leaf=14,
    learning_rate=0.006328507206504974,
    subsample=0.9170443266552768,
    max_features='sqrt',
    random_state=1
)

pipe_gbm = Pipeline(
    steps=[
        ('tree_preprocessor', tree_preprocessor),
        ('regressor2', gbm_tunned)
    ]
)

lgbm_tunned = LGBMRegressor(
    n_estimators=7000,
    max_depth=7,
    learning_rate=0.002536841439596437,
    min_data_in_leaf=22,
    subsample=0.7207500503954922,
    max_bin=210,
    feature_fraction=0.30010067215105635,
    random_state=1,
    verbosity=-1
)

pipe_lgbm = Pipeline(
    steps=[
        ('tree_preprocessor', tree_preprocessor),
        ('regressor3', lgbm_tunned)
    ]
)

catboost_tunned = CatBoostRegressor(
    iterations=4500,
    colsample_bylevel=0.05367479984702603,
    learning_rate=0.018477566955501026, random_strength=0.1321272840705348,
    depth=6,
    l2_leaf_reg=4,
    boosting_type='Plain',
    bootstrap_type='Bernoulli',
    subsample=0.7629052520889268,
    logging_level='Silent',
    random_state=1
)

pipe_catboost = Pipeline(
    steps=[
        ('tree_preprocessor', tree_preprocessor),
        ('regressor4', catboost_tunned)
    ]
)

elasticnet_tunned = ElasticNet(
    max_iter=3993,
    alpha=0.0007824887724782356,
    l1_ratio=0.25,
    tol=3.78681184748232e-06,
    random_state=1
)

pipe_Elasticnet = Pipeline(
    steps=[
        ('linear_preprocessor', linear_preprocessor),
        ('regressor5', elasticnet_tunned)
    ]
)

TargetTransformedElasticnet = TransformedTargetRegressor(
    regressor=pipe_Elasticnet,
    func=np.log1p,
    inverse_func=np.expm1
)

lasso_tunned = Lasso(
    max_iter=2345,
    alpha=0.00019885959230548468,
    tol=2.955506894549702e-05,
    random_state=1
)

pipe_Lasso = Pipeline(
    steps=[
        ('linear_preprocessor', linear_preprocessor),
        ('regressor6', lasso_tunned)
    ]
)

TargetTransformedLasso = TransformedTargetRegressor(
    regressor=pipe_Lasso,
    func=np.log1p,
    inverse_func=np.expm1
)

ridge_tunned = Ridge(
    max_iter=1537,
    alpha=6.654338887411367,
    tol=8.936831872581897e-05,
    random_state=1
)

pipe_Ridge = Pipeline(
    steps=[
        ('linear_preprocessor', linear_preprocessor),
        ('regressor7', ridge_tunned)
    ]
)

TargetTransformedRidge = TransformedTargetRegressor(
    regressor=pipe_Ridge,
    func=np.log1p,
    inverse_func=np.expm1
)

svr_tunned = SVR(
    kernel='linear',
    C=0.019257948556667938,
    epsilon=0.016935170969518305,
    tol=0.0006210492106739069
)

pipe_SVR = Pipeline(
    steps=[
        ('linear_preprocessor', linear_preprocessor),
        ('regressor8', svr_tunned)
    ]
)

TargetTransformedSVR = TransformedTargetRegressor(
    regressor=pipe_SVR,
    func=np.log1p,
    inverse_func=np.expm1
)

'''模型堆叠'''
estimators = [
    ("pipe_xgb", pipe_xgb),
    ("pipe_gbm", pipe_gbm),
    ("pipe_lgbm", pipe_lgbm),
    ("pipe_catboost", pipe_catboost),
    ("TargetTransformedElasticnet", TargetTransformedElasticnet),
    ("TargetTransformedLasso", TargetTransformedLasso),
    ("TargetTransformedRidge", TargetTransformedRidge),
    ("TargetTransformedSVR", TargetTransformedSVR)
]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=Lasso(alpha=0.01, random_state=1))
final_pipe = Pipeline(
    steps=[
        ('stacking_regressor', stacking_regressor),
    ]
)

stacked_regressor = final_pipe.fit(pipe_data, y)
pred_test = stacked_regressor.predict(pipe_test)

output = pd.DataFrame(
    {'Id': pipe_test.index, 'SalePrice': pred_test}
)

output.to_csv('submission.csv', index=False)
