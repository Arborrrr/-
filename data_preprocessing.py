from libs import *


def add_new_features(input_df):
    res = input_df.copy()
    res["Lack_of_feature_index"] = (
            res[
                ["Street", "Alley", "MasVnrType", "GarageType", "MiscFeature", 'BsmtQual', 'FireplaceQu', 'PoolQC',
                 'Fence']
            ].isnull().sum(axis=1) +
            (res["MasVnrType"] == 'None') +
            (res["CentralAir"] == 'No')
    )

    res["MiscFeatureExtended"] = (
            res["PoolQC"].notnull() * 1 + res["MiscFeature"].notnull() * 1 + res["Fence"].notnull() * 1
    ).astype('int64')

    res["Has_Alley"] = res["Alley"].notnull().astype('int64')

    res["Lot_occupation"] = res["GrLivArea"] / res["LotArea"]

    res["Number_of_floors"] = (
            (res["TotalBsmtSF"] != 0).astype('int64') +
            (res["1stFlrSF"] != 0).astype('int64') +
            (res["2ndFlrSF"] != 0).astype('int64')
    )

    res['Total_Close_Live_Area'] = res['GrLivArea'] + res['TotalBsmtSF']

    res['Outside_live_area'] = (
            res['WoodDeckSF'] +
            res['OpenPorchSF'] +
            res['EnclosedPorch'] +
            res['3SsnPorch'] +
            res['ScreenPorch']
    )

    res['Total_usable_area'] = res['Total_Close_Live_Area'] + res['Outside_live_area']

    res['Area_Quality_Indicator'] = res['Total_usable_area'] * res['OverallQual']

    res['Area_Qual_Cond_Indicator'] = (
            res['Total_usable_area'] *
            res['OverallQual'] *
            res['OverallCond']
    )

    res['TotalBath'] = (
            res['FullBath'] +
            (0.5 * res['HalfBath']) +
            res['BsmtFullBath'] +
            (0.5 * res['BsmtHalfBath'])
    )

    res["Has_garage"] = res["GarageYrBlt"].notnull().astype('int64')

    res['House_Age'] = res['YrSold'] - res['YearBuilt']

    res["Is_Remodeled"] = (res["YearBuilt"] != res["YearRemodAdd"]).astype('int64')

    res['HasBsmt'] = res['BsmtQual'].notnull().astype('int64')

    res['Quality_condition'] = res['OverallQual'] * res['OverallCond']

    res['Quality_condition_2'] = res['OverallQual'] + res['OverallCond']

    res['House_Age2'] = res['YrSold'] - res['YearRemodAdd']

    return res


'''读取数据'''
original_train_data = pd.read_csv('input/home-data-for-ml-course/train.csv', index_col='Id')
original_test_df = pd.read_csv('input/home-data-for-ml-course/test.csv', index_col='Id')

'''用于管道中的数据'''
pipe_data = original_train_data.copy()
pipe_test = original_test_df.copy()

'''添加新特征'''
pipe_data = add_new_features(pipe_data)
pipe_test = add_new_features(pipe_test)

'''去除脏数据'''
pipe_data = pipe_data.drop(
    pipe_data[(pipe_data['GrLivArea'] > 4000) & (pipe_data['SalePrice'] < 200000)].index
)

pipe_data = pipe_data.drop(
    pipe_data[(pipe_data['GarageArea'] > 1200) & (pipe_data['SalePrice'] < 300000)].index
)

pipe_data = pipe_data.drop(
    pipe_data[(pipe_data['TotalBsmtSF'] > 4000) & (pipe_data['SalePrice'] < 200000)].index
)

pipe_data = pipe_data.drop(
    pipe_data[(pipe_data['1stFlrSF'] > 4000) & (pipe_data['SalePrice'] < 200000)].index
)

pipe_data = pipe_data.drop(
    pipe_data[(pipe_data['TotRmsAbvGrd'] > 12) & (pipe_data['SalePrice'] < 230000)].index
)

'''确定X、y'''
y = pipe_data.SalePrice
pipe_data = pipe_data.drop("SalePrice", axis=1)

'''对特征进行分类'''
# 类特征
categorical_features = [
    feature for feature in pipe_data.columns if pipe_data[feature].dtype == "object"
]

# 有顺序关系的类特征
ordinal_features = [
    'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
]

# 无顺序关系的类特征
nominal_features = list(set(categorical_features) - set(ordinal_features))

# 数值型特征
numerical_features = list(set(pipe_data.columns) - set(categorical_features))
"""
No_info_columns = ["MoSold", "BsmtFinSF2", "3SsnPorch", "YrSold", "Street", "Condition2", "PoolQC", "Utilities"]
Missing_columns = ["PoolQC", "MiscFeature", "Alley", "Fence"]
Preprocessing_decisions = ["FireplaceQu", "GarageYrBlt", "YearBuilt", "YearRemodAdd"]
"""

'''所有初始管道'''
# Preprocessing for numerical data
numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ]
)

# Preprocessing for categorical data
nominal_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

GarageQual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
Fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
GarageFinish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
KitchenQual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
GarageCond_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
HeatingQC_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
ExterQual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
BsmtCond_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
LandSlope_map = {'Gtl': 2, 'Mod': 1, 'Sev': 0}
ExterCond_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
BsmtExposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
PavedDrive_map = {'Y': 2, 'P': 1, 'N': 0}
BsmtQual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
LotShape_map = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}
BsmtFinType2_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
BsmtFinType1_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
FireplaceQu_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
Utilities_map = {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1, "ELO": 0}
Functional_map = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
PoolQC_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}

ordinal_mapping = [
    {'col': col, 'mapping': globals()[col + '_map']} for col in ordinal_features
]

ordinal_transformer = Pipeline(
    steps=[
        ('ordinal_encoder', ce.OrdinalEncoder(mapping=ordinal_mapping))
    ]
)

# Bundle preprocessing for tree-based algorithms
tree_preprocessor = ColumnTransformer(
    remainder=numerical_transformer,
    transformers=[
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features)
    ]
)

# Preprocessing for numerical data
numerical_transformer2 = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaller', StandardScaler()),
    ]
)

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Kept limited with continuous features
skewed_features = [
    'MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF', 'BsmtFinSF2', 'ScreenPorch', 'EnclosedPorch',
    'Lot_occupation', 'MasVnrArea', 'OpenPorchSF', 'Area_Qual_Cond_Indicator', 'LotFrontage', 'WoodDeckSF',
    'Area_Quality_Indicator', 'Outside_live_area'
]

skewness_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True)),
    ]
)

# Bundle preprocessing for linear algorithms and SVR
linear_preprocessor = ColumnTransformer(
    remainder=numerical_transformer2,
    transformers=[
        ('skewness_transformer', skewness_transformer, skewed_features),
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features),
    ]
)