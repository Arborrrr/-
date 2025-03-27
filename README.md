### 房价预测竞赛的小记录

```libs.py``` 用于引用第三方库  
```data_preprocessing.py``` 用于数据预处理与特征工程  
```tuning.py``` 用于优化模型的超参数(利用 ```optuna``` 库实现)  
```main.py``` 用于最终模型的搭建和预测，将结果保存到 CSV 文件中  
**下面逐步还原项目流程**

### 1. 各个特征的简单描述
| 特征名  | 简单描述  |
|------|------|
| **MSSubClass** | 房屋类型（例如1层、2层房屋等） |
| **MSZoning** | 土地划分类型（例如住宅区、商业区等） |
| **LotFrontage** | 房屋前面的街道长度（以英尺为单位 |
| **LotArea** | 土地面积（以平方英尺为单位） |
| **Street** | 道路类型（例如主街或小街） |
| **Alley** | 小巷类型（如无小巷、铝制小巷等） |
| **LotShape** | 描述土地的形状 |
| **LandContour** | 描述土地的地形状况 |
| **Utilities** | 公共设施类型（如全电、全燃气等） |
| **LotConfig** | 土地配置（如角地、内地等） |
| **LandSlope** | 地形坡度（如平地、陡坡等） |
| **Neighborhood** | 房屋所在的邻里区域 |
| **Condition1** | 房屋附近的道路状况 |
| **Condition2** | 其他道路条件（如果有） |
| **BldgType** | 建筑类型（如独立式、连排等） |
| **HouseStyle** | 房屋样式（如1层、2层等） |
| **OverallQual** | 房屋的整体质量评分（1到10） |
| **OverallCond** | 房屋的整体状况评分（1到10） |
| **YearBuilt** | 房屋建造年份 |
| **YearRemodAdd** | 最后翻修年份 |
| **RoofStyle** | 屋顶类型（如平顶、坡顶等） |
| **RoofMatl** | 屋顶材料（如沥青、木材等） |
| **Exterior1st** | 房屋外部材料的类型（主要材料） |
| **Exterior2nd** | 房屋外部材料的第二种类型（如果有的话） |
| **MasVnrType** | 房屋外墙的面板类型（如石材、砖材等） |
| **MasVnrArea** | 房屋外墙面板的面积（以平方英尺为单位） |
| **ExterCond** | 外部材料状况 |
| **ExterQual** | 外部材料质量 |
| **Foundation** | 基础类型（如混凝土、砖石等） |
| **BsmtQual** | 地下室高度质量 |
| **BsmtCond** | 地下室状况 |
| **BsmtExposure** | 地下室外部暴露程度 |
| **BsmtFinType1** | 地下室的第一个完工类型 |
| **BsmtFinSF1** | 地下室完工部分的面积 |
| **BsmtFinType2** | 地下室的第二个完工类型（如果有） |
| **BsmtFinSF2** | 第二个地下室完工部分的面积 |
| **BsmtUnfSF** | 地下室未完工部分的面积 |
| **TotalBsmtSF** | 地下室的总面积 |
| **Heating** | 供暖类型（如燃气、电力等） |
| **HeatingQC** | 供暖质量和条件 |
| **CentralAir** | 是否有中央空调（Yes/No） |
| **Electrical** | 电力系统类型 |
| **1stFlrSF** | 一楼面积 |
| **2ndFlrSF** | 二楼面积 |
| **LowQualFinSF** | 低质量完工的面积 |
| **GrLivArea** | 地面上生活区域的面积 |
| **BsmtFullBath** | 地下室内完整卫生间的数量 |
| **BsmtHalfBath** | 地下室的半卫生间数量 |
| **FullBath** | 完整卫生间数量 |
| **HalfBath** | 半卫生间数量 |
| **BedroomAbvGr** | 地面层以上的卧室数量 |
| **KitchenAbvGr** | 地面层以上的厨房数量 |
| **KitchenQual** | 厨房质量 |
| **TotRmsAbvGrd** | 地面层以上的总房间数量 |
| **Functional** | 房屋的功能类型（如标准、损坏等） |
| **Fireplaces** | 壁炉数量 |
| **FireplaceQu** | 壁炉质量 |
| **GarageType** | 车库类型（如附加车库、独立车库等） |
| **GarageYrBlt** | 车库建造的年份 |
| **GarageFinish** | 车库完工状态 |
| **GarageCars** | 车库的汽车容量 |
| **GarageArea** | 车库面积 |
| **GarageQual** | 车库质量 |
| **GarageCond** | 车库状况 |
| **PavedDrive** | 道路是否铺设（Yes/No） |
| **WoodDeckSF** | 木质露台的面积 |
| **OpenPorchSF** | 开放式门廊面积 |
| **EnclosedPorch** | 封闭式门廊面积 |
| **3SsnPorch** | 三季门廊面积 |
| **ScreenPorch** | 屏风门廊面积 |
| **PoolArea** | 游泳池面积 |
| **PoolQC** | 游泳池质量 |
| **Fence** | 围栏类型 |
| **MiscFeature** | 其他附加功能（如游泳池、网球场等） |
| **MiscVal** | 其他附加功能的价值 |
| **MoSold** | 销售月份 |
| **YrSold** | 销售年份 |
| **SaleType** | 销售类型（如正常、拍卖等） |
| **SaleCondition** | 销售条件（如正常、延期等） |

### 2. 特征的初步分类

将上述特征进行划分，分为数值特征与分类特征

```python
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
```

进一步将数值特征划分为连续数值特征与离散数值特征

```python
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
```

进一步将分类特征划分为名义特征和顺序特征

```python
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
```

分类后可以进行一个简单的检查，确认一下特征个数是否正确

```python
# 用于检验特征分类是否正确，如果返回 False 将会报错
assert categorical_features.sort() == (nominal_features + ordinal_features).sort()
assert numerical_features.sort() == (discrete_numerical_features + continuous_numerical_features).sort()
```

### 3. 缺失值处理

可以先确认每个特征的缺失值个数，这里对```pandas.DataFrame```类数据使用```.isnull()```方法判断是否为空，空映射到```1```，非空映射到```0```。

```python
# train_df 是用于训练的数据，为 pandas.DataFrame 类
# missing 也为 pandas.DataFrame 类，设定其列标签为 ['count', 'percent']，行标签为全体特征名。
# 其中，'count' 列统计了每个特征的缺失值；'percent' 列统计缺失值在总数据量中的比例，最终按照降序排列，用于直观统计哪些特征的缺失值较多
missing = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False))
missing.columns = ["count"]
missing = missing.loc[(missing!=0).any(axis=1)]
missing["percent"] = missing[0:] / 1460
print(missing.style.background_gradient('viridis'))  # 一种好看的输出格式，不喜欢的话可以直接print(missing)
```

结果显示，特征 ```['PoolQC', 'MiscFeature', 'Alley', 'Fence']``` 的缺失值较多，达到 80% 以上，**可以考虑直接删除这些特征**。
  
对缺失值进行填充，**对全体数值特征，空填充为 ```0```；对分类特征，空填充为 ```'Do_not_have_this_feature'```**   
**这里说明一下，因为该数据中的全体数值特征都是非负的，且均有现实的物理意义，于是可以将空直接替换成```0```*

### 4. 删除偏离数据

绘制数值特征与售价的散点图来观察图像是否具备线性关系，将明显偏离线性关系的数据删除。  
  
**绘制散点图利用```seaborn```库的```scatterplot()```函数实现。**  
参数 ```x=var``` 指定散点图的 ```x``` 轴数据的数据名，可以是一个 ```String```(就是个列名，例如 ```train_df['var']```) 或一个 ```pandas.Series``` 类。  
  
参数 ```y='SalePrice'``` 指定散点图的 ```y``` 轴数据的数据名，是目标特征的名称(```train_df['SalePrice']```)  
  
参数 ```data=train_df``` 指定要绘制的数据集。```train_df``` 是一个 ```pandas.DataFrame```，包含了绘制散点图所需的数据。  
  
```ax=subplot``` 参数用于指定绘图的轴。通常情况下，这个参数与 ```matplotlib``` 的 ```subplots()``` 一起使用，允许你在特定的子图（subplot）中绘制图形。  
  
```hue='SalePrice'``` 参数用于根据某个变量来调节点的颜色。在这里，```hue='SalePrice'``` 表示根据 SalePrice 列的值来设置每个散点的颜色。  

```python
fig, ax = plt.subplots(12, 3, figsize=(23, 60))
for var, subplot in zip(numerical_features, ax.flatten()):
    sns.scatterplot(x=var, y='SalePrice', data=train_df, ax=subplot, hue='SalePrice')
```

其中，可以明显发现特征 ```['1stFlrSF', 'GarageArea', 'GrLivArea', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF']``` 是和售价有线性关系的。进一步观察图像发现除了特征 OverallQual 外，其余特征都有明显的偏离点。下面将对这些点进行删除。

```python
train_df = train_df.drop(
    train_df[
        (train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 200000)
    ].index
)

train_df = train_df.drop(
    train_df[
        (train_df['GarageArea'] > 1200) & (train_df['SalePrice'] < 300000)
    ].index
)

train_df = train_df.drop(
    train_df[
        (train_df['TotalBsmtSF'] > 4000) & (train_df['SalePrice'] < 200000)
    ].index
)

train_df = train_df.drop(
    train_df[
        (train_df['1stFlrSF'] > 4000) & (train_df['SalePrice'] < 200000)
    ].index
)

train_df = train_df.drop(
    train_df[
        (train_df['TotRmsAbvGrd'] > 12) & (train_df['SalePrice'] < 230000)
    ].index
)
```

### 5. 绘制相关系数的热图

这一步是查看数值特征和目标特征的相关性，可以很好的反应线性关系。  
  
**绘制热力图利用 ```seaborn``` 的 ```heatmap()``` 函数来实现，用于可视化相关性矩阵。**  
第一个参数 ```correlation_train``` 是绘制热力图的输入数据，通常是一个二维数组（如 ```pandas.DataFrame```）或矩阵。  
  
参数 ```annot=True``` 用于控制是否在每个单元格中显示数值。如果设置为 ```True```，这里会在热力图的每个小方格中显示相关系数。  
  
参数 ```fmt='.1f'``` 用于控制显示数值的格式。这里的 ```'.1f'``` 指定数值以浮动格式显示，保留 1 位小数。  
  
参数 ```cmap='coolwarm'``` 指定热力图的颜色映射(即色阶)。```'coolwarm'``` 是一个常见的颜色映射方案，通常用于显示有负相关和正相关的数据。颜色从蓝色(代表负相关)到红色(代表正相关)渐变，中间是白色(代表接近零的相关性)。还可以选择其他颜色映射方案，如 ```'viridis', 'inferno', 'Blues'```。  
  
参数 ```square=True``` 用于控制热力图的形状。如果设置为 ```True```，热力图将强制成为一个正方形，使得行数和列数相等时，图形会看起来像一个正方形。绘制相关系数矩阵通常会设定为 ```True```。  
  
参数 ```mask=mask``` 是一个布尔型矩阵，指定哪些值应被隐藏。如果 ```mask``` 是一个与数据矩阵同样形状的布尔数组，```True``` 的位置将隐藏相应的热力图单元格。通常用于遮蔽一些不需要显示的部分，比如在对相关性矩阵进行可视化时，可以选择隐藏上三角矩阵，因为相关性矩阵是对称的。  
  
参数 ```linewidths=1``` 用于控制每个单元格之间的线条宽度。默认线条宽度为 ```0```，如果设置为 ```1```，则会在热力图的单元格之间显示 1 像素的边框线。  
  
参数 ```cbar=False``` 用于控制是否显示颜色条(colorbar)。如果设置为 ```True```，则会显示一个颜色条，用于表示颜色和数据值之间的映射关系。```False``` 则不显示颜色条。

```python
sns.set(font_scale=1.1)
correlation_train = train_df.corr()  # .corr()方法会自动跳过分类特征，无需刻意选择数值特征进行计算
mask = np.triu(correlation_train.corr())  # .triu()方法会创建
plt.figure(figsize=(17, 17))
sns.heatmap(
    correlation_train,
    annot=True,
    fmt='.1f',
    cmap='coolwarm',
    square=True,
    mask=mask,
    linewidths=1,
    cbar=False
)
```

相关系数绝对值较高(>=0.5)的几个数值特征(降序排列)，分别是  
```['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'GarageArea', 'GarageCars', 'YearBulit', 'YearRemodAdd', 'MasVnrArea', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt']```  
可以发现比起画散点图，利用相关系数矩阵更能量化与目标特征的线性关系，可以捕获到更多有价值的特征。

### 6. 计算全部特征的互信息

计算全部特征与目标特征的互信息，可以很好的反应二者的非线性关系。  
  
对于数据特征，需要先填充空值，使用 ```pandas.DataFrame.fillna(0)``` 方法将 ```pandas.DataFrame``` 中全部空值填充为 ```0```。  
  
对于分类特征，可以使用 ```pandas.Series.factorize()``` 方法针对 ```pandas.DataFrame[categorical_features]``` 的每一个特征(```Series```)实现一个到整数集的映射。
  
```.factorize()``` 方法会返回两个值。第一个值是数组，是原数组经过映射后形成的整数数组。  
第二个值是类别列表，包含所有唯一分类值的列表，用于反向映射(这里因为无需反向映射，所以使用 ```_``` 忽略)。  
  
计算互信息使用 ```sklearn.feature_selection.mutual_info_regression()``` 函数实现，需要输入的第一个参数为输入特征(可以是一个 ```DataFrame```)，第二个参数是目标特征(```Serise```)。  
** 要注意数据中不能出现空数据*
  
参数 ```random_state=1``` 是为了确保计算结果的可重复性。  
  
```mutual_info_regression``` 会返回一个与输入特征对应的互信息得分数组。

```python
# 计算数值特征与目标特征的互信息
y = train_df.SalePrice
mutual_df = train_df[numerical_features]
mutual_info = mutual_info_regression(mutual_df.fillna(0), y, random_state=1)
mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["Numerical_Feature_MI"] ).style.background_gradient("cool")

# 计算分类特征与目标特征的互信息
mutual_df_categorical = train_df[categorical_features]
for colname in mutual_df_categorical:
    mutual_df_categorical[colname], _ = mutual_df_categorical[colname].factorize()
mutual_info = mutual_info_regression(mutual_df_categorical.fillna("Do_not_have_feature"), y, random_state=1)  # 其实已经没有空数据了，不用使用 fillna() 填充空数据，这么写只是为了看着统一点。
mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df_categorical.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["Categorical_Feature_MI"] ).style.background_gradient("cool")
```

互信息是越大越好的，可以发现相关系数高的数值特征，互信息也比较高(>0.22)，但也有两个相关系数高但互信息低的特征，如：  
| 特征名  | 互信息  | 相关系数  |
|------|------|------|
| Fireplaces | 0.165573 | 0.5 |
| MasVnrArea | 0.091479 | 0.5 |
  
尤其是特征 MasVnrArea，这可能说明这个特征更容易体现与目标特征的线性关系。  
根据输出结果发现，这些特征```['MoSold', 'BsmtFinSF2', '3SsnPorch', 'YrSold', 'Street', 'Condition2', 'PoolQC', 'Utilities']```的互信息为 0，说明这些特征可能与目标信息呈现完全独立的关系，**可以考虑直接删除这些特征**。

### 7. 添加新特征

**离散型特征(Discrete Features)**  

```python
'''
Lack_of_feature_index: 该特征计算重要特征缺失的数量(["Street", "Alley", "MasVnrType", "GarageType", "MiscFeature",  'BsmtQual', 'FireplaceQu','PoolQC','Fence'])。

MiscFeatureExtended: 该特征计算与房屋相关的一些附加特征(['PoolQC', 'MiscFeature', 'Fence'])的缺失数量。

Has_Alley: 这个特征表示房屋是否有 'Alley'(小巷)。如果 'Alley' 特征非空，则值为 1，表示有小巷，否则为 0。

Number_of_floors: 该特征反应房屋的层数。

Has_garage: 该特征表示房屋是否有车库。如果 'GarageYrBlt' 特征非空，则表示有车库，值为 1，否则为 0。

Is_Remodeled: 该特征表示房屋是否被翻修。如果 'YearBuilt' 与 'YearRemodAdd' 不同，则表示房屋有翻修，值为 1，否则为 0。

TotalBath: 该特征将房屋的各类浴室的总和，半浴室记为0.5。

HasBsmt: 该特征表示房屋是否有地下室。如果 'BsmtQual'(地下室质量) 特征非空，则表示有地下室，值为1，否则为 0。

Quality_conditition_2: 该特征表示房屋的质量 'OverallQual' 和条件 'OverallCond' 之和，用于反映房屋的整体质量和状况。
'''
```
**连续型特征(Continuous Features)**  

```python
'''
Lot_occupation: 该特征表示房屋的建筑面积与土地面积的比例('GrLivArea' / 'LotArea')，用于衡量房屋在地块上的占地密度。

Total_Close_Live_Area: 该特征是房屋的总居住面积，是 'TotalBsmtSF'(地下室面积) 与 'GrLivArea'(主层生活面积) 之和。

Outside_live_area: 该特征表示房屋的户外生活区域，包括'WoodDeckSF'(木甲板)、'OpenPorchSF'(开放阳台)、'EnclosedPorch'(封闭阳台)、'3SsnPorch'(三季阳台) 和 'ScreenPorch'(屏幕阳台)。

Total_usable_area: 该特征是房屋的总可用面积，是 'Total_Close_Live_Area'(总居住面积) 与 'Outside_live_area'(户外生活区域) 之和。

Area_Quality_Indicator: 该特征是 'Total_usable_area'(总可用面积) 与 'OverallQual'(房屋质量) 的乘积，用于衡量房屋的质量与空间的综合影响。

Area_Qual_Cond_Indicator: 该特征是 'Total_usable_area'(总可用面积)、'OverallQual'(房屋质量) 和 'OverallCond'(房屋条件) 的乘积，是考虑了房屋的空间、质量和状况的最综合指标。

House_Age: 该特征表示房屋的总年龄，即 'YearBuilt'(建造年份) 与 'YrSold'(销售年份) 之间的差值。

House_Age2: 该特征表示房屋的翻修年龄，即 'YearRemodAdd'(翻修年份) 与 'YrSold'(销售年份) 之间的差值。

Quality_conditition: 该特征是 'OverallQual'(房屋质量) 与 'OverallCond'(房屋条件) 的乘积，用于对房屋的整体状况进行评分。
'''
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


```
这些特征在互信息中的表现十分良好，于是加入到输入特征中。  

### 8. 管道(Pipeline)制作  

管道有助于避免在状态转换中的数据泄漏(这可能会导致过拟合)，同时它无需手动跟踪训练集和测试集中的预处理步骤，最后可以通过管道来拼接成一个模型用于训练和预测。  
  
下面将把之前各个步骤的特征处理封装为一个个管道。  

**缺失值处理与数据预处理**
```python
# 填充数值特征，将空映射为 0。
numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value = 0))
    ]
)

# 填充名义特征，将空映射为 'Do_not_have_this_feature'，最后再进行 OneHotEncoder 转化为数值型特征。
nominal_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value = 'Do_not_have_this_feature')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# 填充顺序特征，构造一个到整数集的映射，配合 OrdinalEncoder 转化为数字型特征。
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
```

**生成一个树预处理器**

生成一个对树模型进行数据预处理的预处理器，主要是针对三类特征进行编码，保证最终输入树模型的每个特征都是非空且为数值型的。

```python
tree_preprocessor = ColumnTransformer(
    remainder=numerical_transformer,
    transformers=[
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features)
    ]
)
```

**生成一个线性预处理器**

生成一个对线性模型进行数据预处理的预处理器，通过对数值特征进行缩放并处理偏态(skewness)来提高模型的性能，并加快模型训练时的收敛速度。同时使用 ```TransformedTargetRegressor``` 对目标变量进行变换，进一步提高模型的性能。  
  
**偏态（Skewness）** 
  
偏态是统计学中描述数据分布不对称程度的一个度量。它反映了数据分布相对于正态分布（钟形曲线）的偏离程度，具体描述了数据分布的**尾巴**（极端值）向哪一侧倾斜。

**偏态的类型**

1. **正偏态（右偏，Positive Skew）**  
   - **尾巴在右边**：数据的右尾较长，数据集中在较小的值。
   - **偏态值 > 0**：大部分数据点集中在较小的值，少数较大的极端值拉长了右侧的尾巴。
   - **示例**：收入分布通常是正偏态，因为大部分人的收入较低，少数人收入非常高，导致整体分布向右偏斜。

2. **负偏态（左偏，Negative Skew）**  
   - **尾巴在左边**：数据的左尾较长，数据集中在较大的值。
   - **偏态值 < 0**：大部分数据点集中在较大的值，少数较小的极端值拉长了左侧的尾巴。
   - **示例**：考试成绩分布可能表现为负偏态，因为大部分学生的分数较高，而少数学生得分很低。

3. **零偏态（对称，Zero Skew）**  
   - **对称分布**：如果数据是对称的，则偏态值接近0。
   - **示例**：正态分布（钟形曲线）即为零偏态分布。

**偏态的计算**

偏态值通常通过以下公式计算：  

$$
\text{偏态} = \frac{n}{(n-1)(n-2)} \sum \left( \frac{x_i - \bar{x}}{s} \right)^3
$$

- **n**：样本数
- **x_i**：每个数据点
- **x̄**：样本均值
- **s**：样本标准差

- **正偏态**：右尾长，数据集中在左边，偏态值 > 0。
- **负偏态**：左尾长，数据集中在右边，偏态值 < 0。
- **零偏态**：数据对称，偏态值接近0。

**偏态对机器学习的影响**

- 偏态数据对某些机器学习模型（如线性回归、支持向量机等）可能产生影响。大部分模型假设数据近似正态分布，偏态数据可能导致模型拟合效果不佳。
- 为了改善模型效果，可以对偏态数据进行 **变换**（如对数变换、Box-Cox 变换或 Yeo-Johnson 变换），使其更接近正态分布。  
  
先寻找偏态的连续数据特征：

```python
skew_features = train_df[continuous_numerical_features + new_continuous_features].skew().sort_values(ascending=False)

skew_features = pd.DataFrame({'Skew' : skew_features})

skew_features.style.background_gradient('rocket')
```

输出所有值大于1的特征形成一个特征列表，将 skewed_features 更新为该列表。

```python
# 在连续特征中寻找偏态特征，对这些特征进行特殊修正处理
skewed_features = [
    'MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF', 
    'BsmtFinSF2', 'ScreenPorch', 'EnclosedPorch', 'Lot_occupation', 'MasVnrArea', 
    'OpenPorchSF', 'Area_Qual_Cond_Indicator', 'LotFrontage', 'WoodDeckSF',
    'Area_Quality_Indicator', 'Outside_live_area'
]

skewness_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True)),
        # PowerTransformer 通过对数据进行数学变换，使其更接近于正态分布，从而减少偏态。
    ]
)

# 对所有不在偏态特征(skewed_features)中的数值特征进行处理
numerical_transformer2 = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaller', StandardScaler()),  # 用于标准化数值数据，将数值特征调整为均值为0，方差为1的分布
    ]
)

# 对所有分类特征进行处理
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Do_not_have_this_feature')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# 生成针对线性模型和SVR的线性预处理器
linear_preprocessor = ColumnTransformer(
    remainder=numerical_transformer2,
    transformers=[
        ('skewness_transformer', skewness_transformer, skewed_features),
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features),
    ]
)
```

### 9. 模型优化

模型优化使用```optuna```库进行，该库使用起来十分方便，只需规定需要优化的超参数，然后使用交叉验证的方式优化即可，期间还可以结合逐步网格搜索进一步提升优化效率。

### 10. 模型搭建

基础的回归模型可以直接调用第三方库，结合管道可以很方便的将这些基础模型统合成一个大型的模型。  
  
之后就是各种模型的拼接了，使用 staking 统合所有模型的输出结果，将结果作为特征输入到 Lasso 中，然后输出最终结果。












