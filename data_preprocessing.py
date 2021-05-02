import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
#%matplotlib inline
#%%
attrib = read_csv('attributes.csv', delim_whitespace = True)
data = read_csv('communities.data', names = attrib['attributes'])

print(data.shape)
#%%
data.head()
#%%
'''
Remove non-predictive features

state: US state (by number) - not counted as predictive above, but if considered, should be considered nominal (nominal)
county: numeric code for county - not predictive, and many missing values (numeric)
community: numeric code for community - not predictive and many missing values (numeric)
communityname: community name - not predictive - for information only (string)
fold: fold number for non-random 10 fold cross validation, potentially useful for debugging, paired tests - not predictive (numeric)
'''
data = data.drop(columns=['state','county',
                          'community','communityname',
                          'fold'], axis=1)
#%%
data.head()
#%%
'''
Remove column with NA

Some of the features contained many missing values as some surveys were not conducted in some communities, 
so they were removed from the data:
'OtherPerCap', 'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 
'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 
'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 
'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 
'LemasGangUnitDeploy', 'PolicBudgPerPop' 
'''
from pandas import DataFrame

data = data.replace('?', np.nan)
feat_miss = data.columns[data.isnull().any()]
print(feat_miss)

data = data.drop(columns=list(feat_miss), axis=1)
#%%
print(data.shape)
data.head()
#%%
data.describe()
#%%
# ViolentCrimesPerPop: total number of violent crimes per 100K popuation (numeric - decimal)
# GOAL attribute (to be predicted)
data.hist(column = ['ViolentCrimesPerPop'], bins = 30, color = 'red', alpha = 0.8)
plt.show()

#%%
# TODO　Correlations
import seaborn as sns

corrmat = data.corr()
fig = plt.figure(figsize = (16, 12))

sns.heatmap(corrmat, vmax = 0.8)
plt.show()
#%%
corrT = data.corr(method = 'pearson').round(4)
corrT = corrT.sort_values(by=['ViolentCrimesPerPop'])
corrT_VCPP = corrT['ViolentCrimesPerPop']
#%%
'''
Remove Multicollinearity
set VIF = 5, R^2 = 0.8 to remove attributes
'''
'''Dimensionality Reduction - Principal Component Analysis (PCA) 
The dataset contain many variables highly 
correlated. Multicolinearity will increase the model variance. Dimensionality reduction utilizing PCA can provide an 
optimal set of orthogonal features. Let's adopt the criterion in which we select those principal components 
responsible to explain more than a unit variance ("eigenvalue one criterion"). '''
X_DF = data.iloc[:, 0:99]
# data.to_csv("data_removed.csv")
# Detecting Multicollinearity using VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X_DF):
    # X_DF = pd.DataFrame(X)
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X_DF.columns
    vif["VIF"] = [variance_inflation_factor(X_DF.values, i) for i in range(X_DF.shape[1])]

    return(vif)

VIF = calc_vif(X_DF)

#%%
data_to_dump = VIF.where(VIF['VIF'] > 30)
data_to_dump = data_to_dump.dropna(how='any')
columns_to_dump = list(data_to_dump.iloc[:, 0])
X_DF = data.drop(columns=columns_to_dump, axis=1)

#%%
# VIF_2 = calc_vif(X_DF)

'''
Now we have two racePct*** remain, consider corrT_VCPP['racePctAsian'] = 0.0376, corrT_VCPP['racePctHisp'] = 0.2931,
which means racePctAsian is not very related to ViolentCrimesPerPop, so to simplify
the model, we only keep racePctWhite as our sensitive variable.
'''
X_DF = X_DF.drop(columns=['racePctAsian', 'racePctHisp'], axis=1)

print("Removed columns(", len(columns_to_dump) + 2, "):\n", (columns_to_dump + ['racePctAsian', 'racePctHisp']))
#%%
from sklearn.model_selection import train_test_split

X = X_DF.values
y = data.iloc[:, 99].values

seed = 0

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)

print(X.shape)
print(y.shape)

#%%
from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%%
# Perform PCA
# from sklearn.decomposition import PCA
#
# c = 14
# pca = PCA(n_components = c)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
#
# print("Amount of variance: %s" % pca.explained_variance_)
# print("Sum of the variance: %s" % sum(pca.explained_variance_).round(2))
#
# print("Percentage of variance: %s" % pca.explained_variance_ratio_)
# print("Sum of the percentage of variance: %s" % sum(pca.explained_variance_ratio_).round(2))
#
#
# plt.scatter(np.arange(1,(c+1)),pca.explained_variance_, c = 'red')
# plt.plot((0,15), (1,1), color = 'black', linestyle = 'dashed')
# plt.xlabel('PC')
# plt.ylabel('Amount of variance explained')
# plt.show()
# print(X_train.shape)
#%%
pd.DataFrame(X_train).to_csv('X_train.csv')
pd.DataFrame(X_test).to_csv('X_test.csv')
pd.DataFrame(y_train).to_csv('y_train.csv')
pd.DataFrame(y_test).to_csv('y_test.csv')
pd.DataFrame(X).to_csv('X.csv')
pd.DataFrame(y).to_csv('y.csv')