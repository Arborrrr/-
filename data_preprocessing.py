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
# 数值型特征
numerical_features = [
    "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
    "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"
]

# 分类特征
categorical_features = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "ExterQual",
    "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
    "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "Fence", "MiscFeature", "PoolQC",
    "SaleType", "SaleCondition"
]

# 连续数值特征
continuous_numerical_features = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'MiscVal', 'YrSold'
]

# 离散数值特征
discrete_numerical_features = [
    'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
    'GarageCars', 'MoSold', "MSSubClass"
]

# 顺序特征
ordinal_features = [
    'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
]

# 名义特征
nominal_features = [
    "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "CentralAir", 'Electrical', "GarageType",
    "MiscFeature", "SaleType", "SaleCondition"
]

# 新添加的连续数值特征
new_continuous_features = [
    "Lot_occupation", 'Total_Close_Live_Area', 'Outside_live_area',
    'Total_usable_area','Area_Quality_Indicator', 'House_Age',
    'Area_Qual_Cond_Indicator', 'Quality_conditition', 'House_Age2'
]

# 新添加的离散数值特征
new_discrete_features = [
    "Lack_of_feature_index", "MiscFeatureExtended", "Has_Alley",
    "Number_of_floors", "Has_garage", "Is_Remodeled",
    'TotalBath', 'HasBsmt', 'Quality_conditition_2'
]

# 更新数值特征
numerical_features = numerical_features + new_continuous_features + new_discrete_features

# 更新连续数值特征
continuous_numerical_features = continuous_numerical_features + new_continuous_features

# 更新离散数值特征
discrete_numerical_features = discrete_numerical_features + new_discrete_features

"""
No_info_columns = ["MoSold", "BsmtFinSF2", "3SsnPorch", "YrSold", "Street", "Condition2", "PoolQC", "Utilities"]
Missing_columns = ["PoolQC", "MiscFeature", "Alley", "Fence"]
Preprocessing_decisions = ["FireplaceQu", "GarageYrBlt", "YearBuilt", "YearRemodAdd"]
"""

'''所有初始管道'''
# 将数值特征的空数据填充为 0
numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ]
)

# 将名义特征的空数据填充为 'Do_not_have_this_feature'，再进行独热编码
nominal_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# 顺序特征的映射组
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

# 全体顺序特征的映射
ordinal_mapping = [
    {'col': col, 'mapping': globals()[col + '_map']} for col in ordinal_features
]

# 对顺序特征进行有序映射
ordinal_transformer = Pipeline(
    steps=[
        ('ordinal_encoder', ce.OrdinalEncoder(mapping=ordinal_mapping))
    ]
)

# 创建树预处理管道
tree_preprocessor = ColumnTransformer(
    remainder=numerical_transformer,
    transformers=[
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features)
    ]
)

# 将数据特征的空数据填充为 0，再进行规范化，使数据趋近标准正态分布
numerical_transformer2 = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaller', StandardScaler()),
    ]
)

# 对全体分类特征进行空值填充，再进行独热编码
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# 偏度特征
skewed_features = [
    'MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF',
    'BsmtFinSF2', 'ScreenPorch', 'EnclosedPorch', 'Lot_occupation', 'MasVnrArea',
    'OpenPorchSF', 'Area_Qual_Cond_Indicator', 'LotFrontage', 'WoodDeckSF',
    'Area_Quality_Indicator', 'Outside_live_area'
]

# 将偏度特征的空数据映射为 0，在进行PowerTransformer尽可能消除偏度
skewness_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True)),
    ]
)

# 建立为线性模型准备的线性预处理管道
linear_preprocessor = ColumnTransformer(
    remainder=numerical_transformer2,
    transformers=[
        ('skewness_transformer', skewness_transformer, skewed_features),
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features),
    ]
)