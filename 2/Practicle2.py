import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
def detect_outliers(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    outliers = df[(df[var] < low) | (df[var] > high)]
    print(f"{var} - Outliers detected: {len(outliers)}")
    
    return df[(df[var] >= low) & (df[var] <= high)]

df = pd.read_csv(r"acdemic_data.csv")
#---------------------------------------------------------------------------------------
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset: \n', df.isna().sum())
#---------------------------------------------------------------------------------------
# Fill the missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['SPOS'].fillna(df['SPOS'].mean(), inplace=True)
print('Total Number of Null Values in Dataset: \n', df.isna().sum())

df['DSBDA'] = pd.to_numeric(df['DSBDA'], errors='coerce')
df['DSBDA'] = df['DSBDA'].fillna(df['DSBDA'].mean())  # or dropna if preferred

df['DA'] = pd.to_numeric(df['DA'], errors='coerce')
df['DA'] = df['DA'].fillna(df['DA'].mean())  # or dropna if preferred

#---------------------------------------------------------------------------------------
# Converting categorical to numeric using Find and replace method
df['Gender'].replace(['F','M'],[0,1],inplace=True)

#---------------------------------------------------------------------------------------
# Outliers can be visualized using boxplot
# using seaborn library we can plot the boxplot
fig, axes = plt.subplots(2,2)
fig.suptitle('Before removing Outliers')
sns.boxplot(data = df, x ='SPOS', ax=axes[0,0])
sns.boxplot(data = df, x ='DSBDA', ax=axes[0,1])
sns.boxplot(data = df, x ='WT', ax=axes[1,0])
sns.boxplot(data = df, x ='DA', ax=axes[1,1])
plt.show()
#Display and remove outliers
df = detect_outliers(df, 'SPOS')
df = detect_outliers(df, 'SPOS')
df = detect_outliers(df, 'WT')
df = detect_outliers(df, 'DSBDA')
df = detect_outliers(df, 'DSBDA')
fig, axes = plt.subplots(2,2)
fig.suptitle('After removing Outliers')
sns.boxplot(data = df, x ='SPOS', ax=axes[0,0])
sns.boxplot(data = df, x ='DSBDA', ax=axes[0,1])
sns.boxplot(data = df, x ='WT', ax=axes[1,0])
sns.boxplot(data = df, x ='DA', ax=axes[1,1])
plt.show()
#---------------------------------------------------------------------------------------
print('---------------- Data Skew Values before Yeo John Transformation ----------------------')
# There are two types
# 1. Left skew
# 2. Right skew
# Formula to find out data skewness = 3*(mean-median)/std
# = 0 (no skew) print
# = negative (Negative skew) left skewed data
# = positve (Positive skew) Right skewed data
# = -0.5 to 0 to 0.5 (acceptable skew)
# = -0.5> <-1 moderate negative skew
# = 0.5> <1 moderate positive skew
# = > -1 high negative
# = > 1 high positive
print('SPOS: ', df['SPOS'].skew())
print('DSBDA: ', df['DSBDA'].skew())
print('WT: ', df['WT'].skew())
print('DA: ', df['DA'].skew())
fig, axes = plt.subplots(2,2)
fig.suptitle('Handling Data Skewness')
sns.histplot(ax = axes[0,0], data = df['SPOS'], kde=True)
sns.histplot(ax = axes[0,1], data = df['WT'], kde=True)
from sklearn.preprocessing import PowerTransformer
yeojohnTr = PowerTransformer(standardize=True)
df['SPOS'] = yeojohnTr.fit_transform(df['SPOS'].values.reshape(-1,1))
df['Discussion'] = yeojohnTr.fit_transform(df['WT'].values.reshape(-1,1))
print('---------------- Data Skew Values after Yeo John Transformation ----------------------')
print('SPOS: ', df['SPOS'].skew())
print('WT: ', df['WT'].skew())
sns.histplot(ax = axes[1,0], data = df['SPOS'], kde=True)
sns.histplot(ax = axes[1,1], data = df['WT'], kde=True)
plt.show()