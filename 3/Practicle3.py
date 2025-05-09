import pandas as pd
import numpy as np
df = pd.read_csv(r"acdemic_data.csv")  # Update path if needed
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
#df['Gender'].replace(['F','M'],[0,1],inplace=True)

#---------------------------------------------------------------------------------------
# groupwise statistical information
print('Groupwise Statistical Summary....')
print('\n-------------------------- SPOS -----------------------\n')
print(df['SPOS'].groupby(df['Gender']).describe())
print('\n-------------------------- DSBDA -----------------------\n')
print(df['DSBDA'].groupby(df['Gender']).describe())
print('\n-------------------------- WT -----------------------\n')
print(df['WT'].groupby(df['Gender']).describe())
#---------------------------------------------------------------------------------------

