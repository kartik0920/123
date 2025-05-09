# Importing libraries
import pandas as pd
import numpy as np
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv(r"StudentsPerformance.csv")
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
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Fill the missing values
df['math score'].fillna(df['math score'].mode()[0], inplace=True)
df['reading score'].fillna(df['reading score'].mean(), inplace=True)
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# changing data type of columns
# see the datatype using df.dtypes
# change the datatype using astype
df['math score']=df['math score'].astype(int)
print('Change in datatype: ', df['math score'].dtypes)

#---------------------------------------------------------------------------------------
# Converting categorical (qualitative) variable to numeric (quantitative) variable

# 1. Find and replace method
# 2. Label encoding method
# 3. OrdinalEncoder using scikit-learn

# Find and replace method
df['gender'].replace(['Female','Male'],[0,1],inplace=True)
# Label encoding method
df['test preparation course']=df['test preparation course'].astype('category') #change data type to category
df['test preparation course']=df['test preparation course'].cat.codes

print('After converting categorical variable to numeric variable: ')
print(df.head().T)
#---------------------------------------------------------------------------------------
# Normalization of data
# converting the range of data into uniform range
# marks [0-100] [0-1]
# salary [200000 - 200000 per month] [0-1]
# Min-max feature scaling
# minimum value = 0
# maximum value = 1
# when we design model the higher value over powers in the model
df['math score']=(df['math score']-df['math score'].min())/(df['math score'].max()-df['math score'].min())
# (x - min value into that column)/(max value - min value)
# Maximum absolute scaler using scikit-learn
from sklearn.preprocessing import MaxAbsScaler
abs_scaler=MaxAbsScaler()
df[['writing score']]=abs_scaler.fit_transform(df[['writing score']])
#---------------------------------------------------------------------------------------
print(df.head().T)
