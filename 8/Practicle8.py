# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
# Reading dataset
df = sns.load_dataset('titanic')
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
# Display and fill the Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
df['age'].fillna(df['age'].median(), inplace=True)
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Single variable histogram
fig, axis = plt.subplots(1,3)
sns.histplot(ax = axis[0], data = df, x='sex', hue = 'sex', multiple = 'dodge', shrink = 0.8)
sns.histplot(ax = axis[1], data = df, x='pclass', hue = 'pclass',multiple = 'dodge', shrink = 0.8)
sns.histplot(ax = axis[2], data = df, x='survived', hue = 'survived', multiple = 'dodge', shrink = 0.8)
plt.show()
# Single variable histogram
fig, axis = plt.subplots(1,2)
sns.histplot(ax = axis[0], data = df, x='age', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1], data = df, x='fare', multiple = 'dodge', shrink = 0.8, kde = True)
plt.show()
# Two variable histogram
fig, axis = plt.subplots(2,2)
sns.histplot(ax = axis[0,0], data = df, x='age', hue = 'sex', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[0,1], data = df, x='fare', hue = 'sex', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,0], data=df, x='age', hue = 'survived', multiple = 'dodge',shrink=0.8, kde= True)
sns.histplot(ax = axis[1,1], data=df, x='fare', hue='survived', multiple='dodge', shrink=0.8, kde = True)
plt.show()
# Two variable histogram
fig, axis = plt.subplots(2,2)
sns.histplot(ax=axis[0,0], data=df, x='sex', hue='survived', multiple= 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[0,1], data=df, x='pclass', hue='survived', multiple='dodge', shrink=0.8, kde= True)
sns.histplot(ax=axis[1,0], data=df, x='age', hue='survived', multiple='dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,1], data=df, x='fare', hue='survived', multiple='dodge', shrink = 0.8, kde = True)
plt.show()